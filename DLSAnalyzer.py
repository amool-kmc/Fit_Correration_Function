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
#波長(nm)
LAMBDA = 632
#屈折率
REFRACTIVE_INDEX = 1.33
#ボルツマン定数
BOLTZMANN_CONST = 1.380649 * 10**(-23) 
#ASCファイルにおけるデータ行の開始と終わり(27,217デフォルト)
ASC_DATA_START_ROW = 40
ASC_DATA_END_ROW = 217
#ウインドウサイズ
WINDOW_SIZE = (5.5,4.0)

def main():
    '''
    メイン関数

    解析テンプレート:
    taus, qs, temps = analysisfordir("完全パス")
    show_graph(temps, tau*10**3, xlabel="temperature [℃]", ylabel="relaxation time [ms]", title="tau vs temp")
    show_tauinv_t_graph(temp, tau, fitting=False, tc=34)
    '''
    

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

def calculating_q(angle):
    '''
    散乱ベクトルを計算する関数

    Parameters
    ----------
    angle : float
        DLS装置での角度パラメーターの値
    
    Returns
    -------
    q : float
        散乱ベクトルの大きさ
    '''
    theta = angle * 18 / 6000 #6000 = 18°
    return 4 * np.pi * REFRACTIVE_INDEX * np.sin(np.deg2rad(theta)/2) / (LAMBDA * (10**-9)) #散乱ベクトルの大きさ

def averaging(data_array,x_delta,x_min,x_max):
    '''
    yの平均を求める関数
    デルタのあり方に注意。（誤差がでる）

    Parameters
    ----------
    data_array : array
        使用例:
        # array1 = [x_array1, y_array1]
        # array1 = [x_array2, y_array2]
        # array3 = [x_array3, y_array3]
        # data_array = [array1,array2,array3]
        # x_array, avl_y_array = averaging(data_array, dx, minx, maxx)
        
        要素はnp配列。同じxについてyを平均化

    Returns
    -------
    x_array : ndarray
    value_avl : ndarray
    '''
    iterator = np.zeros(len(data_array),dtype='int')
    x_array = np.empty(0)
    value_avl = np.empty(0)

    x = x_min
    while(x < x_max):
        match_num = 0
        value_sum = 0
        # 各データセットを走査
        for i,array in enumerate(data_array):
            # 各データセットについてxの存在をチェック
            if(x - x_delta/2 <= array[0][iterator[i]] and array[0][iterator[i]] <= now_x + x_delta/2):
                value_sum += array[1][iterator[i]]
                match_num += 1
                if(iterator[i] != len(array[0])-1):
                    iterator[i] += 1
        
        if(match_num > 0):
            value_avl = np.append(value_avl,value_sum/match_num)
            x_array = np.append(x_array,x)

        x += x_delta
        print(x)

    return x_array,value_avl

def calculating_particle_size(taus,qs,temps):
    '''
    粒子サイズを求める関数。VV測定の結果に対して用いることができる。

    Parameters
    ----------
    taus : ndarray
        緩和時間（s）
    temps : ndarray
        温度（摂氏）
    qs : ndaaray
        散乱ベクトルの大きさ
    
    Returns
    -------
    particle_size : ndarray
        粒子の流体力学半径（m）
    '''
    # 温度をケルビンに直す
    temps = temps + 273.15
    # 粘度（20℃の水がおよそ10**-3）
    eta = 0.89 * 10**(-3)

    # 並進拡散定数
    D = 1 / (2 * taus * qs**2)

    return (BOLTZMANN_CONST * temps) / (6 * np.pi * eta * D)

###--------------------------------------------------------
###自己相関関数の規格化
###--------------------------------------------------------
def nomalize(y_data):
    #平均から最大値を算出
    max_cont = 50
    max_tot = 0
    for i in range(max_cont):
        max_tot += y_data[i]
    max_ave = max_tot / max_cont

    #平均から最小値を算出
    min_cont = 50
    min_tot = 0
    for i in range(min_cont):
        min_tot += y_data[-i]
    min_ave = min_tot / min_cont

    #規格化
    new_y = (y_data - min_ave) / (max_ave - min_ave)

    return new_y

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

def show_graph(xdata,ydata,xlabel="x",ylabel="y",title=""):
    '''
    二次元グラフを表示。

    Parameters
    ----------
    xdata : nparray
        横軸データの配列
    ydata : nparray
        縦軸データの配列
    xlabel : string
        横軸ラベル
    ylabel : string
        縦軸ラベル
    title : string
        タイトル
    '''
    ###グラフの生成
    fig = plt.figure(figsize=WINDOW_SIZE)
    ax = fig.add_subplot(111)

    fig.suptitle(title, size=18, weight=2)
    #散布図

    #グラフの描画
    print(len(xdata))
    print(len(ydata))
    ax.scatter(xdata,ydata,marker='o',s=20)
    #グラフの範囲指定
    ax.set_xlim([30,40])
    ax.set_ylim([0,50])

    #目盛り文字サイズ
    ax.tick_params(labelsize = 10)

    #グラフのセット
    ax.grid(True,linestyle="--")
    #ax.legend(fontsize=10,title="sample")
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)

    plt.show()

def show_tau_t_graph(temps,taus):
    '''
    緩和時間vs温度のグラフを表示。

    Parameters
    ----------
    temps : nparray
        温度の配列
    taus : nparray
        緩和時間配列

    '''
    ###グラフの生成
    fig = plt.figure()
    ax = fig.add_subplot(111)
    fig.suptitle("tau vs temperature", size=18, weight=2)
    #散布図

    #グラフの描画
    ax.scatter(temps,taus*1000,marker='o',s=20)
    

    #グラフのセット
    ax.grid(True)
    #ax.legend(fontsize=10,title="sample")
    #目盛り文字サイズ
    ax.tick_params(labelsize = 10)
    ax.set_xlabel('temperature', fontsize=10)
    ax.set_ylabel('relaxation time[ms]', fontsize=10)

    plt.show()

def show_tauinv_t_graph(temps,taus,fitting=False,tc=30.0):
    '''
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

    Returns
    -------
    -b/a : float
        臨界温度
    '''

    ###グラフの生成
    fig = plt.figure(figsize=WINDOW_SIZE)
    ax = fig.add_subplot(111)
    fig.suptitle("tauinv vs temperature", size=18, weight=2)
    #生データグラフのセット
    tauinvs = 1 / (taus*1000)
    ax.scatter(temps,tauinvs,marker='o',s=20)

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

        #転移点を探し（二分木）、フィッティング線の描く範囲を決定
        tc_error = 0.01
        topside = 50
        underside = 10
        while(1):
            serchpoint = (topside + underside)/2
            if(liner_function(serchpoint,a,b)<0 and liner_function(serchpoint + tc_error,a,b)>0):
                x_axis = np.linspace(serchpoint,temps[-1],1000)
                break
            elif(liner_function(serchpoint + tc_error,a,b) < 0):
                underside = serchpoint + tc_error
            elif(liner_function(serchpoint,a,b) > 0):
                topside = serchpoint

        y_fit = liner_function(x_axis,a,b)
        ax.plot(x_axis, y_fit, linewidth=2)

        return -b/a


    #グラフの範囲指定
    ax.set_xlim([30,40])
    ax.set_ylim([0,1])

    #グラフ設定と表示
    ax.grid(True)
    ax.legend(fontsize=10,title="sample")
    #目盛り文字サイズ
    ax.tick_params(labelsize = 10)
    ax.set_xlabel('temperature [℃]', fontsize=10)
    ax.set_ylabel('relaxation frequency [1/ms]', fontsize=10)

    plt.show()

    
def analysisfordir(inputFileDirectory):
    '''
    指定ディレクトリ内のASCファイルを解析し、自己相関関数のフィッティングを行う。

    Parameters
    ----------
    inputFileDirectory : string
        データASCファイルの入っているディレクトリへの完全パス。

    Returns
    -------
    taus : ndarray
        緩和時間(s)の配列
    qs : ndarray
        散乱ベクトルの配列
    temps : ndarray
        温度の配列
    '''

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
            print(temperature," fitting completed")

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
        temp_label = str('{:.3g}'.format(temperature/10))
        angle_label = str('{:.3g}'.format(angle*18./6000.))
        ax_corf.scatter(x_data,y_data,marker='o',s=2,label=temp_label)
        
        #フィッティング曲線
        #x軸刻み(最小オーダー、最大オーダー、プロット数)
        if(plot_fittingcurve):
            x_axis = np.logspace(-3.2,4.2,1000)
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