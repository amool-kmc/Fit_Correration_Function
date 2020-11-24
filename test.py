import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import os
import glob
import re

# *****************************
# ASCファイル名 命名規則
# (角度パラメーター値)_(摂氏温度x10)_(偏光板配置)_(測定時間)_(サンプルの名前).ASC
# *****************************


###定数定義
#波長[nm]
LAMBDA = 532
REFRACTIVE_INDEX = 1.33
#ASCファイルにおけるデータ行の開始と終わり
ASC_DATA_START_ROW = 27
ASC_DATA_END_ROW = 209


###--------------------------------------------------------
###メイン関数
###--------------------------------------------------------
def main():
    x = np.linspace(20,40,21)
    y = np.array([500,500,500,500,500,500,500,500,500,500,562,579,400,310,250,130,10,7,5,2,1])
    show_tauinv_t_graph(x,y,fitting=True,tc=30)
    
###--------------------------------------------------------
###フィッティング関数の定義
###--------------------------------------------------------
def correlationFunction(x,a,b,beta,tau):
    return a*np.exp(-(x/tau)**beta) + b

def liner_function(x,a,b):
    return a*x + b

###--------------------------------------------------------
###二重緩和のとき小さい緩和をカットする関数（参照渡し）
###--------------------------------------------------------
def cutdata(y_data,threshold):
    for i in range(len(y_data)):
        if(y_data[i] > threshold):
            y_data[i] = y_data[i] - threshold
        elif(y_data[i] <= threshold):
            y_data[i] = 0

###--------------------------------------------------------
###散乱ベクトルを計算する関数
###--------------------------------------------------------
def calculating_q(angle):
    theta = angle * 18 / 6000 #6000 = 18°
    return 4 * np.pi * REFRACTIVE_INDEX * np.sin(np.deg2rad(theta)/2) / (LAMBDA * (10**-9)) #散乱ベクトルの大きさ

###--------------------------------------------------------
###自己相関関数の規格化
###--------------------------------------------------------
def nomalize(y_data):
    #平均から最大値を算出
    max_cont = 30
    max_tot = 0
    for i in range(max_cont):
        max_tot += y_data[i]
    max_ave = max_tot / max_cont

    #平均から最小値を算出
    min_cont = 30
    min_tot = 0
    for i in range(min_cont):
        min_tot += y_data[-i]
    min_ave = min_tot / min_cont

    #規格化
    new_y = (y_data - min_ave) / (max_ave - min_ave)

    return new_y

###--------------------------------------------------------
###ASCファイルからデータを抽出し、NumPy配列で返す関数
###--------------------------------------------------------
def getdata(input_data_file):
    '''
    ASCファイルからデータを抽出し、データをNumPy配列で返す関数

    Parameters
    ----------
    input_data_file : string
        ASCファイルの絶対パス
    
    Returns
    -------
    npdata : nparray
    '''
    #ASCファイルを開く
    datafile = open(input_data_file)

    #内容を行区切りでリストとして取得
    rawlist = datafile.readlines()

    #return用の配列
    npdata = np.empty((0,2), float)

    #データを抽出
    for i,rawline in enumerate(rawlist):
        if(ASC_DATA_START_ROW <= i and i <= ASC_DATA_END_ROW): #データ行の指定（30秒なら27~217）
            #データ行をtab区切りで分割しNumPy配列に
            npline = np.array(rawline.split(), dtype = float)

            #2次元NumPy配列として格納（[[time, value]]）
            npdata = np.append(npdata, [npline], axis = 0)

    datafile.close()

    return npdata

def show_tau_t_graph(temps,taus):
    """
    緩和時間vs温度のグラフを表示。

    Parameters
    ----------
    temps : nparray
        温度の配列
    taus : nparray
        緩和時間配列

    """
    ###グラフの生成
    fig = plt.figure()
    ax = fig.add_subplot(111,title="tau vs temperature")

    #散布図

    #グラフの描画
    ax.scatter(temps,taus,marker='o',s=8)

    #グラフのセット
    ax.grid(True)
    ax.legend(fontsize=10,title="sample")
    ax.set_xlabel('temperature', fontsize=12)
    ax.set_ylabel('relaxation time[ms]', fontsize=12)

    plt.show()

def show_tauinv_t_graph(temps,taus,fitting=False,tc=30.0):
    """
    緩和周波数vs温度のグラフを表示。オプションでフィッティングが可能。

    Parameters
    ----------
    temps : ndarray
        温度の配列
    taus : ndarray
        緩和時間配列
    fitting : bool
        フィッティングの有無（デフォルトでFalse）
    tc : float
        フィッティングの際の温度の閾値。これ以上のデータでフィッティングを行う。（デフォルトで30℃）

    """
    ###グラフの生成
    fig = plt.figure()
    ax = fig.add_subplot(111,title="tau vs temperature")

    #生データグラフのセット
    tauinvs = 1 / taus
    ax.scatter(temps,tauinvs,marker='o',s=8)
    
    if(fitting):
        for i,t in enumerate(temps):
            if(t>tc):
                x_data = temps[i:]
                y_data = tauinvs[i:]
                break
        
        #フィッティング
        init_parameter = [0,0]
        param_opt, cov = curve_fit(liner_function,x_data,y_data,init_parameter)

        a = param_opt[0]
        b = param_opt[1]

        x_axis = np.linspace(temps[0],temps[-1],1000)
        y_fit = liner_function(x_axis,a,b)
        ax.plot(x_axis, y_fit, linewidth=1.3)


    #グラフ設定と表示
    ax.grid(True)
    ax.legend(fontsize=10,title="sample")
    ax.set_xlabel('temperature', fontsize=12)
    ax.set_ylabel('relaxation time[ms]', fontsize=12)

    plt.show()

def analysisfordir(inputFileDirectory):
    """
    指定ディレクトリ内のASCファイルを解析し、自己相関関数のフィッティングを行う。

    Parameters
    ----------
    inputFileDirectory : string
        データASCファイルの入っているディレクトリへの完全パス。

    Returns
    -------
    taus : ndarray
        緩和時間の配列
    qs : ndarray
        散乱ベクトルの配列
    temps : ndarray
        温度の配列
    """

    ###ファイル読み込み-----------------------------------------
    #データASCファイルへのパスをすべて読み込み
    input_data_files = glob.glob(os.path.join(inputFileDirectory,"*.ASC"))

    #ファイルの順番を温度でソート、配列の番号を変えれば他のパラメーターでもソート可能
    input_data_files.sort(key=lambda s: int(re.sub(".ASC","",re.sub(inputFileDirectory + "\\\\","",s)).split("_")[0]))

    #結果格納用の配列
    taus = np.empty(0)
    qs = np.empty(0)
    temps = np.empty(0)

    #グラフインスタンス生成
    fig_corf = plt.figure()
    ax_corf = fig_corf.add_subplot(111,title="Correlataion Function")

    #各ASCについて解析-------------------------------------------------------------
    for input_data_file in input_data_files:
        #ファイル名から各パラメーターを取得（angle_temp_closs_time_name.ASC）
        filename = re.sub(".ASC","",re.sub(inputFileDirectory + "\\\\","",input_data_file)) #ファイル名取り出し（パスの部分と拡張子を除去）
        input_parameters = filename.split("_") #パラメータを取得

        #取得パラメータ
        angle = float(input_parameters[0])
        temperature = float(input_parameters[1])
        time = float(input_parameters[3])
        sample_name = input_parameters[4]
        
        #データセット
        data = getdata(input_data_file)
        x_data = data[:,0]
        y_data = data[:,1]
        init_parameter = [1,0,1,1]

        #データの規格化
        y_data = nomalize(y_data)
        
        #フィッティング曲線表示のフラグ
        plot_fittingcurve = True

        ###自己相関関数のフィッティング-----------------------------------------------
        #フィッティングが失敗したときは例外処理を施し、生データのみプロットする
        try:
            param_opt, cov = curve_fit(correlationFunction,x_data,y_data,init_parameter)
            print(angle," fitting completed")

            #フィッティングパラメーターの取得
            a = param_opt[0]
            b = param_opt[1]
            beta = param_opt[2]
            tau = param_opt[3]

            #τの保存
            taus = np.append(taus,tau * (10**-3))
            #角度の計算、保存
            q = calculating_q(angle)
            qs = np.append(qs,q)
            #温度の保存
            temps = np.append(temps,temperature/10)

        except RuntimeError:
            print(angle," fitting failed")
            plot_fittingcurve = False


        

        ###グラフの表示----------------------------------------------------
        #生データ
        ax_corf.scatter(x_data,y_data,marker='o',s=2,label=str('{:.3g}'.format(angle*18./6000.)))
        
        #フィッティング曲線
        #x軸刻み(最小オーダー、最大オーダー、プロット数)
        if(plot_fittingcurve):
            x_axis = np.logspace(-4.2,4.2,1000)
            y_fit = correlationFunction(x_axis,a,b,beta,tau)
            ax_corf.plot(x_axis, y_fit, linewidth=1.3)
        
        ###ラベル等の設定
        ax_corf.grid(True)
        ax_corf.legend(fontsize=10,title="angle")
        ax_corf.set_ylim([-0.1,1.3])
        ax_corf.set_xlabel('time (ms)', fontsize=12)
        ax_corf.set_ylabel('f', fontsize=12)

        #logscaleに
        setting1 = plt.gca()
        setting1.set_xscale('log')

        
    plt.show()
    
    return taus, qs, temps


###--------------------------------------------------------------------------------------
if(__name__ == '__main__'):
    main()