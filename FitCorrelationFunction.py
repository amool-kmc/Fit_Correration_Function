import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
import pandas as pd
import os
import glob
import re

###定数定義
#波長
LAMBDA = 532
#ASCファイルにおけるデータ行の開始と終わり
ASC_DATA_START_ROW = 27
ASC_DATA_END_ROW = 217


###--------------------------------------------------------
###メイン関数
###--------------------------------------------------------
def main():
    analysisfordir("/KUniv/Q10/191129")
    #analysisfordir("/KUniv/Q10/191029asc/32deg")
    #analysisfordir("/KUniv/Q10/191029asc/34deg")

    plt.show()

###--------------------------------------------------------
###フィッティング関数の定義
###--------------------------------------------------------
def correlationFunction(x,a,b,beta,tau):
    return a*np.exp(-(x/tau)**beta) + b

def tau_qFunction(x,a,b):
    return a*x + b

###--------------------------------------------------------
###散乱ベクトルを計算する関数
###--------------------------------------------------------
def calculating_q(angle):
    theta = angle * 18 / 6000 #6000 = 18°
    return 4 * np.pi * np.sin(np.deg2rad(theta)/2) / LAMBDA #散乱ベクトルの大きさ

###--------------------------------------------------------
###ASCファイルからデータを抽出し、NumPy配列で返す関数
###--------------------------------------------------------
def getdeta(inputDataFile):
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

    #グラフの描画
    plt.plot(qs,taus_inv,'o',markersize=4)

    #フィッティング曲線
    #x軸刻み(最小オーダー、最大オーダー、プロット数)
    q_axis = np.linspace(-5.7,-4.8,1000)
    tau_fit = tau_qFunction(q_axis,param_opt[0],param_opt[1])
    plt.plot(q_axis, tau_fit, linewidth=1.3, label=inputFileDirectory)
    plt.legend("")

    #グラフのセット
    plt.grid(True)
    plt.title("tau-q")
    plt.xlabel('log(q)', fontsize=12)
    plt.ylabel('log(1/tau)', fontsize=12)
    plt.text(-5.6,3.5,"grad  " + str(param_opt[0]))
    #setting2 = plt.gca()
    #setting2.set_xscale('log')
    #setting2.set_yscale('log')

    #返り値は傾き
    return param_opt[0]
    

###--------------------------------------------------------
###指定ディレクトリ内のASCファイルを解析し、自己相関関数のフィッティングを行い、各種解析をする
###--------------------------------------------------------
def analysisfordir(inputFileDirectory):
    ###ファイル読み込み-----------------------------------------
    #データASCファイルへのパスをすべて読み込み
    inputDataFiles = glob.glob(os.path.join(inputFileDirectory,"*.ASC"))

    #結果格納用の配列
    taus = np.empty(0)
    qs = np.empty(0)

    #各ASCについて解析-------------------------------------------------------------
    for inputDataFile in inputDataFiles:
        #ファイル名から各パラメーターを取得（angle_temp_closs_time_name.ASC）
        filename = re.sub(".ASC","",re.sub(inputFileDirectory + "\\\\","",inputDataFile)) #ファイル名取り出し（パスの部分と拡張子を除去）
        input_parameters = filename.split("_") #パラメータを取得
        
        #取得パラメータ
        angle = float(input_parameters[0])
        temperature = float(input_parameters[1])
        time = float(input_parameters[3])

        #ある角度では弾くようにするときはここで
        if(angle >= 10000):
            continue
        qs = np.append(qs,calculating_q(angle))


        #データセット
        data = getdeta(inputDataFile)
        x_data = data[:,0]
        y_data = data[:,1]
        init_parameter = [1,0,1,1]

        ###自己相関関数のフィッティング-----------------------------------------------
        param_opt, cov = curve_fit(correlationFunction,x_data,y_data,init_parameter)

        #フィッティングパラメーターの取得
        a = param_opt[0]
        b = param_opt[1]
        beta = param_opt[2]
        tau = param_opt[3]

        taus =np.append(taus,tau)


        ###グラフの表示----------------------------------------------------
        #'''
        #生データ
        plt.plot(x_data,y_data,'o',label="raw data",markersize=4)

        #フィッティング曲線
        #x軸刻み(最小オーダー、最大オーダー、プロット数)
        x_axis = np.logspace(-4.2,4.2,1000)
        y_fit = correlationFunction(x_axis,a,b,beta,tau)
        plt.plot(x_axis, y_fit,linewidth=1.3, label='fitted curve')

        #ラベル等の設定
        plt.grid(True)
        plt.title("Correlataion Function")
        #plt.legend(loc='upper right')
        plt.xlabel('time (ms)', fontsize=12)
        plt.ylabel('I', fontsize=12)

        #グラフの描画
        setting1 = plt.gca()
        setting1.set_xscale('log')
        #plt.show()

        #グラフの保存
        #plt.savefig(os.path.join(pngDirectory,))
        


        #自己相関関数の表示
        plt.show()
        #'''

    #tau-qグラフ生成
    tauqgraph(taus,qs,inputFileDirectory)


###--------------------------------------------------------------------------------------
main()