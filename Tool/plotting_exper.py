from matplotlib import pyplot as plt
import seaborn as sns
import numpy as np
import os
from PyPDF2 import PdfFileMerger
from sklearn.manifold import TSNE
import pandas as pd
from graphviz import Digraph

def Save_stock1_SP500(split_series, indices, stock_name, index, p_pre, rmse, r_next, Label_trans, root):
    dot = Digraph(comment='The Round Table')
    dot.graph_attr["rankdir"] = "LR"
    dot.attr(label='Regime-switching of ' + str(stock_name) + ' series', labelloc='top')
    edge_list = []
#
    for i in range(len(indices) - 1):
        # dot.node(chr(ord('a') + i), 'R' + str(Label_trans[i, index] + 1))  r'$W_1$'
        dot.node(chr(ord('a') + i), 'R' + str(Label_trans[i, index] + 1))
        edge_list.append(chr(ord('a') + i) + chr(ord('a') + i + 1))

    dot.node(chr(ord('a') + len(indices) - 1), 'R' + str(r_next + 1), color='red', fontcolor='red')

    dot.edges(edge_list)

    dot.render(root + 'results/stock1/SP500_RS_images/' + stock_name,
               view=False, format='pdf')



    plt.figure(figsize=(17, 3))
    plt.plot(p_pre, color='b', label='Predicted time series')
    plt.plot(split_series[len(indices) - 1].reset_index(drop=True).iloc[:, index], color='gray', linestyle='--',
             label='True time series')
    plt.title(stock_name + ', RMSE =' + str(rmse))
    plt.legend(loc='upper right')
    plt.savefig(
        root + 'results/stock1/SP500_predict_images/' + stock_name + '.pdf',
        dpi=600,
        format='pdf',
        bbox_inches='tight')
    plt.show()
    plt.close()


def stock1_examples(series, split_series, indices, P_value_all, RMSE):

    Stock_example_list = ['MRK', 'MU', 'ETSY']
    fig = plt.figure(figsize=(15, 6))
    for i in range(len(Stock_example_list)):
        stock_name = Stock_example_list[i]
        index = series.columns.get_loc(stock_name)
        ax = fig.add_subplot(len(Stock_example_list), 1, i + 1)
        ax.plot(P_value_all.iloc[index, :], color='white', label=stock_name + ', RMSE =' + str(RMSE[index]))
        ax.plot(P_value_all.iloc[index, :], color='b', label='Predicted time series')
        ax.plot(split_series[len(indices) - 1].reset_index(drop=True).iloc[:, index], color='gray', linestyle='--',
                label='True time series')

        ax.tick_params(labelsize=12)
        ax.legend(loc='upper right', fontsize=17, framealpha=0.7)
    fig.tight_layout()
    plt.show()


def stock2_examples(series, Pre_all_win, True_all_win):
    Stock_example_list = ['CSX', 'ULTA', 'UNP', 'BK']
    series_re = series.reset_index(drop=True)
    series_re = series_re.T.reset_index(drop=True).T
    Pre_all_win_re = Pre_all_win.set_index(Pre_all_win.index.values + 44, drop=True)
    index = [series.columns.get_loc(Stock_example_list[e]) for e in range(len(Stock_example_list))]
    fig = plt.figure(figsize=(15, 8))
    colors = ['blue', 'green', 'red', 'c']
    ax1 = fig.add_subplot(2, 1, 1)
    for i in range(len(index)):
        ax1.plot(series_re.iloc[:, index[i]], color=colors[i], label=Stock_example_list[i], linewidth=1.2)
    ax1.set_xlim(xmin=0, xmax=142)
    ax1.set_xlabel('Time', fontsize=15)
    ax1.set_ylabel('Value', fontsize=15)
    ax1.tick_params(labelsize=12)
    ax1.legend(fontsize=15)
    ax1.axvline(66, color='red', linestyle='--')
    ax1.axvline(77, color='red', linestyle='--')
    ax1.axvline(99, color='red', linestyle='--')
    ax1.axvline(110, color='red', linestyle='--')
    ax2 = fig.add_subplot(2, 1, 2)
    for i in range(len(index)):
        ax2.plot(Pre_all_win_re.iloc[:, index[i]], color=colors[i], label=Stock_example_list[i], linewidth=1.2)
    ax2.set_xlim(xmin=0, xmax=142)
    ax2.set_xlabel('Time', fontsize=15)
    ax2.set_ylabel('Value', fontsize=15)
    ax2.tick_params(labelsize=12)
    ax2.legend(fontsize=15)
    ax2.axvline(44, color='gray', linestyle='--')
    ax2.axvline(66, color='red', linestyle='--')
    ax2.axvline(77, color='red', linestyle='--')
    ax2.axvline(99, color='red', linestyle='--')
    ax2.axvline(110, color='red', linestyle='--')
    plt.show()

def stock2_sub_examples(series, Pre_all_win):
    Stock_example_list = ['CSX', 'ULTA', 'UNP', 'BK']
    series_re = series.reset_index(drop=True)
    series_re = series_re.T.reset_index(drop=True).T
    Pre_all_win_re = Pre_all_win.set_index(Pre_all_win.index.values + 44, drop=True)
    index = [series.columns.get_loc(Stock_example_list[e]) for e in range(len(Stock_example_list))]
    fig = plt.figure(figsize=(6, 4))
    colors = ['blue', 'green', 'red', 'c']
    ax1 = fig.add_subplot(2, 1, 1)
    for i in range(len(index)):
        ax1.plot(series_re.iloc[66:77, index[i]], color=colors[i], label=Stock_example_list[i], linewidth=1.2)
    # ax1.set_xlim(xmin=0)
    # ax1.set_xlabel('Time',fontsize=10)
    ax1.set_ylabel('Value', fontsize=10)
    ax1.tick_params(labelsize=8)

    ax2 = fig.add_subplot(2, 1, 2)
    for i in range(len(index)):
        ax2.plot(Pre_all_win_re.iloc[22:33, index[i]], color=colors[i], label=Stock_example_list[i], linewidth=1.2)
    # ax2.set_xlim(xmin=0)
    ax2.set_ylim(ymax=np.max(np.max(series_re.iloc[66:77, index], axis=0)))
    ax2.set_xlabel('Time', fontsize=10)
    ax2.set_ylabel('Value', fontsize=10)
    ax2.tick_params(labelsize=8)

    plt.show()

def Save_SP500(split_series, indices, stock_name, index, p_pre, rmse, r_next, Label_trans, root, slide_point,
               start_slide):
    dot = Digraph(comment='The Round Table')
    dot.graph_attr["rankdir"] = "LR"
    edge_list = []

    for i in range(len(indices) - 1):
        dot.node(chr(ord('a') + i), 'R' + str(Label_trans[i, index] + 1))
        edge_list.append(chr(ord('a') + i) + chr(ord('a') + i + 1))

    dot.node(chr(ord('a') + len(indices) - 1), 'R' + str(r_next + 1), color='red', fontcolor='red')

    dot.edges(edge_list)

    dot.render(root + 'results/stock2/SP500_RS_images/' + 'Slide_point_' + str(
        slide_point - start_slide + 1) + '/' + 'Slide_' + str(
        slide_point - start_slide + 1) + '_' + stock_name,
               view=False, format='pdf')


    plt.figure(figsize=(17, 3))
    plt.plot(p_pre, color='b', label='Predicted time series')
    plt.plot(split_series[len(indices) - 1].reset_index(drop=True).iloc[:, index], color='gray', linestyle='--',
             label='True time series')

    plt.title(stock_name + ', RMSE =' + str(rmse))
    plt.legend(loc='upper right')
    plt.savefig(
        root + 'results/stock2/SP500_predict_images/' + 'Slide_point_' + str(
            slide_point - start_slide + 1) + '/' + 'Slide_' + str(
            slide_point - start_slide + 1) + '_' + stock_name + '.pdf',
        dpi=600,
        format='pdf',
        bbox_inches='tight')
    plt.show()
    plt.close()


def save_r(R, cluster, slide_point, root, start_slide):
    fig = plt.figure(figsize=(15, 2))
    R = R.reset_index(drop=True)
    for i in range((R.T).shape[1]):
        ax = fig.add_subplot(1, len(cluster), i + 1)
        ax.plot((R.T).iloc[:, i], color='black')
    plt.title('Regime of Slide_point ' + str(slide_point))
    plt.savefig(
        root + 'results/stock2/Regime_images/' + 'Slide_point' + str(slide_point - start_slide + 1) + '.pdf',
        dpi=600,
        format='pdf',
        bbox_inches='tight')
    plt.show()
    plt.close()


def save_stock1_rs(Label_trans, R, root):
    R_s = np.zeros((len(R), len(Label_trans)))
    for i in range(len(Label_trans)):
        a, b = np.unique(Label_trans[i, :], return_counts=True)
        for j in range(len(a)):
            R_s[a[j], i] = b[j]
    R_s = pd.DataFrame(R_s.astype(int))
    R_s.columns = [r'$W_1$', r'$W_2$', r'$W_3$', r'$W_4$', r'$W_5$', r'$W_6$']
    R_s.index = [r'$R_1$', r'$R_2$', r'$R_3$', r'$R_4$', r'$R_5$']
    fig = plt.figure(figsize=(15, 2))
    sns.heatmap((R_s).astype(int), annot=True, annot_kws={'size': 20}, fmt='.0f', xticklabels=True, linewidths=1,
                linecolor='black',
                yticklabels=True,
                square=False, cmap="YlGn", cbar=False, cbar_kws={"shrink": .39})
    plt.yticks(rotation=0)
    plt.savefig(
        root + 'results/stock1/Stock1_RS' + '.pdf',
        dpi=600,
        format='pdf',
        bbox_inches='tight')
    plt.show()


def regime_iden(R, cluster):
    fig = plt.figure(figsize=(15, 2))

    R = R.reset_index(drop=True)
    colors = ['blue', 'gold', 'lime', 'red', 'k']
    for i in range(R.T.shape[1]):
        ax = fig.add_subplot(1, len(cluster), i + 1)
        ax.plot(R.T.iloc[:, i], label='Regime' + str(i + 1), color=colors[i])
        ax.legend(loc='upper right', fontsize=18)
        ax.tick_params(labelsize=15)
    fig.tight_layout()
    plt.show()


def tsne(Label_trans, split_series, root):
    fig = plt.figure(figsize=(15, 3))
    # plt.title('t-SNE visualization of window')
    for i in range(len(Label_trans)):
        ax = fig.add_subplot(1, len(Label_trans), i + 1)
        x = split_series[i].T
        y = Label_trans[i, :]
        tsne = TSNE(n_components=2, verbose=0, random_state=123)
        z = tsne.fit_transform(x)
        df = pd.DataFrame()
        df["y"] = y
        df["y"] = df["y"].replace(0, r'$R_1$')
        df["y"] = df["y"].replace(1, r'$R_2$')
        df["y"] = df["y"].replace(2, r'$R_3$')
        df["y"] = df["y"].replace(3, r'$R_4$')
        df["y"] = df["y"].replace(4, r'$R_5$')
        df[r'$D_1$'] = z[:, 0]
        df[r'$D_2$'] = z[:, 1]
        dict = {0: r'$R_1$', 1: r'$R_2$', 2: r'$R_3$', 3: r'$R_4$', 4: r'$R_5$'}
        colors = {r'$R_1$': 'blue', r'$R_2$': 'gold', r'$R_3$': 'lime', r'$R_4$': 'red', r'$R_5$': 'k'}

        markers1 = []
        for j in range(len(np.unique(y))):
            markers1.append('+')

        sns.scatterplot(x=r'$D_1$', y=r'$D_2$', hue=df.y.to_list(), style=df.y.to_list(), markers=markers1,
                        palette=colors,
                        data=df).set(title=r'$W_%s$' % str(i + 1))
        current_handles, current_labels = plt.gca().get_legend_handles_labels()
        a = []
        for j in range(len(current_labels)):
            a.append(list(dict.keys())[list(dict.values()).index(current_labels[j])])
        b = sorted(a)
        current_labels = [dict[b[e]] for e in range(len(b))]
        current_handles = [current_handles[a.index(b[e])] for e in range(len(b))]
        ax.legend(current_handles, current_labels, fontsize=13)
        plt.ylabel(r'$D_2$', rotation='horizontal', fontsize=13)
        plt.xlabel(r'$D_1$', fontsize=13)

    fig.tight_layout()
    plt.show()



def merge_allpre(Pre_all_win, True_all_win, series, root):
    Stock_name_list = series.columns.values.tolist()
    for i in range(np.shape(Pre_all_win)[1]):
        stock_name = Stock_name_list[i]
        index = series.columns.get_loc(stock_name)

        plt.figure(figsize=(17, 3))
        plt.plot(Pre_all_win.iloc[:, index], color='b', label='Predicted time series')
        plt.plot(True_all_win.iloc[:, index], color='gray', linestyle='--',
                 label='True time series')

        plt.title(stock_name)
        plt.legend(loc='upper right')
        plt.savefig(
            root + 'results/stock2/SP500_predict_images/' + 'All_window' + '/' + stock_name + '.pdf',
            dpi=600,
            format='pdf',
            bbox_inches='tight')
        plt.show()
        plt.close()

    target_path = root + 'results/stock2/SP500_predict_images/All_window/'
    pdf_lst = [f for f in os.listdir(target_path) if f.endswith('.pdf')]
    pdf_lst = [os.path.join(target_path, filename) for filename in pdf_lst]

    file_merger = PdfFileMerger()
    for pdf in pdf_lst:
        file_merger.append(pdf)

    file_merger.write(
        root + 'results/stock2/' + 'Merge_all_win_pre.pdf')
    file_merger.close()

