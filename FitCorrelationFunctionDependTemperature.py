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
    
    
    taus1,qs1,temps1 = analysisfordir("/KUniv/Q10/200220/agalyovh_temp25")
    
###--------------------------------------------------------
###フィッティング関数の定義
###--------------------------------------------------------
def correlationFunction(x,a,b,beta,tau):
    return a*np.exp(-(x/tau)**beta) + b

def tau_qFunction(x,a,b):
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
def getdata(inputDataFile):
    #ASCファイルを開く
    datafile = open(inputDataFile)

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


###---------------------------------------------------------------------------
###1/tauとqのlogを取り、そのグラフを描いて（インスタンス生成のみ、plotはしない）傾きを調べる関数
###---------------------------------------------------------------------------
def tauqgraph(taus,qs,inputFileDirectory):
    #1/tau,qのlogを取る
    taus_inv = np.log(1 / taus)
    qs = np.log(qs)

    #フィッティング
    init_parameter = [2,10]
    param_opt, cov = curve_fit(tau_qFunction,qs,taus_inv,init_parameter)

    ###グラフの生成
    fig_tq = plt.figure()
    ax_tq = fig_tq.add_subplot(111,title="ln(1/tau) vs ln(q)")

    #生データのグラフの描画
    ax_tq.scatter(qs,taus_inv,marker='o',s=8)
    
    #フィッティング曲線の描画
    #x軸刻み(最小オーダー、最大オーダー、プロット数)
    q_axis = np.linspace(15,16,1000)
    tau_fit = tau_qFunction(q_axis,param_opt[0],param_opt[1])
    ax_tq.plot(q_axis, tau_fit, c="red", linewidth=1.3, label=inputFileDirectory)

    #グラフのセット
    ax_tq.grid(True)
    ax_tq.set_xlabel('log(q)', fontsize=12)
    ax_tq.set_ylabel('log(1/tau)', fontsize=12)
    fig_tq.text(0.13,0.9,"gradient  " + str('{:.3g}'.format(param_opt[0])))
    
    #返り値は傾き
    return param_opt[0]
    
###---------------------------------------------------------------------------
###1/tauとqのlogを取り、そのグラフを重ねて描く関数（引数はnp.arrayのnp.array([[] []])）
###---------------------------------------------------------------------------
def draw_multi_tauqgraph(tauss,qss,label,title,leg_title):
    ###グラフの生成
    fig_tq = plt.figure()
    ax_tq = fig_tq.add_subplot(111,title=title)

    #散布図と曲線の準備
    for i in range(len(tauss)):
        taus = tauss[i]
        qs = qss[i]

        #1/tau,qのlogを取る
        taus_inv = np.log(1 / taus)
        qs = np.log(qs)

        #フィッティング
        init_parameter = [2,10]
        param_opt, cov = curve_fit(tau_qFunction,qs,taus_inv,init_parameter)


        #生データのグラフの描画
        ax_tq.scatter(qs,taus_inv,marker='o',s=8,label=label[i])
        
        #フィッティング曲線の描画
        #x軸刻み(最小オーダー、最大オーダー、プロット数)
        q_width = calculating_q(12000)-calculating_q(6000)
        q_axis = np.linspace(np.log(calculating_q(6000)-q_width*0.05), np.log(calculating_q(12000)+q_width*0.05), 1000)
        tau_fit = tau_qFunction(q_axis,param_opt[0],param_opt[1])
        ax_tq.plot(q_axis, tau_fit, linewidth=1.3)

    #グラフのセット
    ax_tq.grid(True)
    ax_tq.legend(fontsize=10,title=leg_title)
    ax_tq.set_xlim([np.log(calculating_q(6000)-q_width*0.05), np.log(calculating_q(12000)+q_width*0.05)])
    ax_tq.set_ylim([3.5,7])
    ax_tq.set_xlabel('log(q)', fontsize=12)
    ax_tq.set_ylabel('log(1/tau)', fontsize=12)
    #fig_tq.text(0.13,0.9,"gradient  " + str('{:.3g}'.format(param_opt[0])))

    plt.show()
        
###---------------------------------------------------------------------------
###緩和時間(tau)と温度の関数
###---------------------------------------------------------------------------
def tau_tgraph(taus,temps):
    ###グラフの生成
    fig = plt.figure()
    ax = fig.add_subplot(111,title="tau vs temperature")

    #散布図

    #生データのグラフの描画
    ax.scatter(temps,taus,marker='o',s=8)

    #グラフのセット
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
    taus : nparray
        緩和時間の配列
    qs : nparray
        散乱ベクトルの配列
    temps : nparray
        温度の配列
    """

    ###ファイル読み込み-----------------------------------------
    #データASCファイルへのパスをすべて読み込み
    inputDataFiles = glob.glob(os.path.join(inputFileDirectory,"*.ASC"))
    #温度でソート
    inputDataFiles.sort(key=lambda s: int(re.sub(".ASC","",re.sub(inputFileDirectory + "\\\\","",s)).split("_")[0]))
    print(inputDataFiles)

    #結果格納用の配列
    taus = np.empty(0)
    qs = np.empty(0)
    temps = np.empty(0)

    #グラフインスタンス生成
    fig_corf = plt.figure()
    ax_corf = fig_corf.add_subplot(111,title="Correlataion Function")

    #各ASCについて解析-------------------------------------------------------------
    for inputDataFile in inputDataFiles:
        #ファイル名から各パラメーターを取得（angle_temp_closs_time_name.ASC）
        filename = re.sub(".ASC","",re.sub(inputFileDirectory + "\\\\","",inputDataFile)) #ファイル名取り出し（パスの部分と拡張子を除去）
        input_parameters = filename.split("_") #パラメータを取得

        
        
        #取得パラメータ
        angle = float(input_parameters[0])
        temperature = float(input_parameters[1])
        time = float(input_parameters[3])
        sample_name = input_parameters[4]
        

        #データセット
        data = getdata(inputDataFile)
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