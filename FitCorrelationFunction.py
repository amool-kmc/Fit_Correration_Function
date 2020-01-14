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
    taus1,qs1 = analysisfordir("/KUniv/Q10/200114/agalyo_vh_fix")
    taus2,qs2 = analysisfordir("/KUniv/Q10/200114/lyoonly_vh")
    #analysisfordir("/KUniv/Q10/191029asc/32deg")
    #analysisfordir("/KUniv/Q10/191029asc/34deg")

    #tau-qグラフを重ねて描画
    tauss = np.array([taus1,taus2])
    qss = np.array([qs1,qs2])
    label = ["agalyo","lyo"]
    draw_multi_tauqgraph(tauss,qss,label)


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

    ###グラフの生成
    fig_tq = plt.figure()
    ax_tq = fig_tq.add_subplot(111,title="ln(1/tau) vs ln(q)")

    #生データのグラフの描画
    ax_tq.scatter(qs,taus_inv,marker='o',s=8)
    
    #フィッティング曲線の描画
    #x軸刻み(最小オーダー、最大オーダー、プロット数)
    q_axis = np.linspace(-5.7,-4.8,1000)
    tau_fit = tau_qFunction(q_axis,param_opt[0],param_opt[1])
    ax_tq.plot(q_axis, tau_fit, c="red", linewidth=1.3, label=inputFileDirectory)

    #グラフのセット
    ax_tq.grid(True)
    #plt.title("ln(1/tau) vs q")
    ax_tq.set_xlabel('log(q)', fontsize=12)
    ax_tq.set_ylabel('log(1/tau)', fontsize=12)
    fig_tq.text(0.13,0.9,"gradient  " + str('{:.3g}'.format(param_opt[0])))
    
    #返り値は傾き
    return param_opt[0]
    
###---------------------------------------------------------------------------
###1/tauとqのlogを取り、そのグラフを重ねて描く関数（引数はnp.arrayのnp.array([[] []])）
###---------------------------------------------------------------------------
def draw_multi_tauqgraph(tauss,qss,label):
    ###グラフの生成
    fig_tq = plt.figure()
    ax_tq = fig_tq.add_subplot(111,title="ln(1/tau) vs ln(q)")

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
        q_axis = np.linspace(-5.7,-4.8,1000)
        tau_fit = tau_qFunction(q_axis,param_opt[0],param_opt[1])
        ax_tq.plot(q_axis, tau_fit, linewidth=1.3)

    #グラフのセット
    ax_tq.grid(True)
    ax_tq.legend(fontsize=10,title="sample")
    ax_tq.set_xlabel('log(q)', fontsize=12)
    ax_tq.set_ylabel('log(1/tau)', fontsize=12)
    #fig_tq.text(0.13,0.9,"gradient  " + str('{:.3g}'.format(param_opt[0])))

    plt.show()
        
        


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

    #グラフインスタンス
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
        data = getdeta(inputDataFile)
        x_data = data[:,0]
        y_data = data[:,1]
        init_parameter = [1,0,1,1]
        
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

            #τの追加
            taus = np.append(taus,tau)
            #角度の計算、追加
            q = calculating_q(angle)
            qs = np.append(qs,q)

        except RuntimeError:
            print(angle," fitting failed")
            plot_fittingcurve = False


        

        ###グラフの表示----------------------------------------------------
        #生データ
        ax_corf.scatter(x_data,y_data,marker='o',s=1.5,label=str('{:.3g}'.format(angle*18./6000.)))
        
        #フィッティング曲線
        #x軸刻み(最小オーダー、最大オーダー、プロット数)
        if(plot_fittingcurve):
            x_axis = np.logspace(-4.2,4.2,1000)
            y_fit = correlationFunction(x_axis,a,b,beta,tau)
            ax_corf.plot(x_axis, y_fit, linewidth=1.3)
        
        ###ラベル等の設定
        ax_corf.grid(True)
        ax_corf.legend(fontsize=10,title="angle")
        ax_corf.set_ylim([0,1])
        ax_corf.set_xlabel('time (ms)', fontsize=12)
        ax_corf.set_ylabel('I', fontsize=12)

        #logscaleに
        setting1 = plt.gca()
        setting1.set_xscale('log')

        
    
    plt.show()
    #tau-qグラフ生成
    #tauqgraph(taus,qs,inputFileDirectory)

    return taus, qs


###--------------------------------------------------------------------------------------
if(__name__ == '__main__'):
    main()