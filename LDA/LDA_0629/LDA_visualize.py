import pandas as pd
import pyLDAvis

fn = { "LDA_pickle_n": "pkl/LDA_{n}.pickle" ,
      "热点图png": "html_out/热点图_seaborn{suffix}.png",
       "热点图svg": "html_out/热点图_seaborn{suffix}.svg",
       "热点图htm": "html_out/热点图_plotly{suffix}.html",
       "柱状图htm": "html_out/热点图_plotly{suffix}.html", #{N}_{kind}_{info}
      }

#-----------------------------------------------------------------------------------
# 1. 数据处理 仿區塊建模 blockmodelling

# 未过阈值设零
# 使用前用此代码去做阈值设置
# simplify.__defaults__ = (0, 0.33)
def simplify(x, threshold = 0.66):    
    if x > threshold:
        return(x)
    else:
        return(0)
    
def simplify_matrix(m, threshold):
    simplify.__defaults__ = (0, threshold) # 使用前用此代碼去做閾值設置
    ms = m.applymap(simplify)\
          .sort_values(by=list(m.columns), ascending=False)
    return(ms)

# 仿區塊建模 blockmodelling A
def sort_bythreshold_w_topic(m, threshold, topic_label="主题"):    
    ms = simplify_matrix (m, threshold)
    # ms.index  ms.columns  按ms 去排序 m
    # m = m.loc[ms.index, ms.columns]  # 全排
    m = m.loc[:, ms.columns]           # 排ms.columns
    
    m_主题 = m.copy()  #
    m_主题.columns = [topic_label+str(x) for x in m_主题.columns ]
    m_主题
    return(m_主题)

# 仿區塊建模 blockmodelling B
def easy_block_single(m, threshold, i, debug=False):
    # input- doc-topic matrix: m
    # output- sorted doc-topic matrix by blocks
    ms = simplify_matrix (m, threshold)
    ms_block_br = ms.loc[:,m.columns[i]].argmin()  # break for blocks, .argmin()最小值的索引
    
    upper = ms[:ms_block_br].sort_values(by=list(m.columns)[i+1:])
    lower = ms[ms_block_br:].sort_values(by=list(m.columns)[i+1:], ascending=False)#
    ms_sorted = pd.concat([upper,lower])
    
    if debug==True:
        display ("ms_block_br", ms_block_br)
        display ("upper",upper)
        display ("lower",lower)
    
    m = m.loc[ms_sorted.index, ms_sorted.columns]
    return(m)

def easy_block(m, threshold, debug=False):
    _range_ = list(range(m.shape[1]-1)) # list(range(m.shape[1]-2,-1,-1))
    _range_.reverse()
    #print (_range_)
    for i in _range_:
        #print ("i", i)
        m = easy_block_single(m, threshold, i, debug=debug)
    return(m)

#-----------------------------------------------------------------------------------
# 2. 数据可视化 seaborn
import matplotlib.pyplot as plt
import seaborn as sns  # seaborn

zh_font = "Microsoft YaHei" # "MingLiU"#"SimSun"# "Microsoft JhengHei"#"DFKai-SB" "FangSong"#'DengXian'#'Microsoft YaHei'
plt.rcParams['font.sans-serif'] = [zh_font]  # 中文字体设置-黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题
plt.rcParams['xtick.labelsize'] = 24  
plt.rcParams['ytick.labelsize'] = 24 
sns.set(font=zh_font)  # 解决Seaborn中文显示问题

def visualize_seaborn(data_input, 
                      suffix="全", 
                      figsize=(14,18), 
                      color_map="YlGnBu"):# 配色 可選不一樣的 "YlGnBu"  "coolwarm"
    plt.figure(figsize=figsize)
    ax = sns.heatmap(data_input, cmap=color_map, annot=True, \
                     fmt="0.3f", linewidth =.5 )
    fig = ax.get_figure()
    fig.subplots_adjust(left=0.35)
    fig.savefig(fn["热点图png"].format(suffix=suffix), dpi=300)
    fig.savefig(fn["热点图svg"].format(suffix=suffix), dpi=300)

    
# 3. 数据可视化 plotly
import plotly.express as px        # 取px.colors.sequential.Blues
import plotly.offline as offline   # 取offline
import plotly.graph_objects as go
import plotly.figure_factory as ff

def visualize_plotly_heatmap(data_input, suffix="全", \
                             title = '文本-主题交叉分析：交互式热点图', \
                             label_x_tpl = "主题{x}", label_y_tpl = "文本{y}", \
                             figsize=(1800,2000), color_map=px.colors.sequential.Blues):
    data_input = data_input.iloc[::-1] # .iloc[::-1] reverse order for heatmap
    layout1 = go.Layout(\
        title = title,\
        font = dict(family='SimHei, DengXian, "Microsoft YaHei"', size=24),\
        autosize = False, width=figsize[0], height=figsize[1],\
                   margin=dict( l=50, r=50, b=100, t=100,pad=4),\
        )

    # fig1
    z = data_input.to_numpy()
    display(z)
    z_text = data_input.applymap(lambda x:"{x:0.3f}".format(x=x)).applymap(lambda x:x.replace("0.000","")).to_numpy()
    x = [label_x_tpl.format(x=x) for x in list(data_input.columns)] 
    display(x)
    y = [label_y_tpl.format(y=y) for y in list(data_input.index)]  
    display(y)

    fig1 = ff.create_annotated_heatmap(z=z, x=x, y=y, annotation_text=z_text,\
                                    font_colors=['darkblue', 'white'],\
                                    colorscale = color_map, \
                                    )
    fig1.update_layout(layout1)    

    filename=fn["热点图htm"].format(suffix = suffix)
    offline.plot(fig1, filename=filename)


def visualize_plotly_bar(data_input, suffix="全", \
                             title = '文本-主题交叉分析：交互式热点图', \
                             label_x_tpl = "主题{x}", label_y_tpl = "文本{y}", \
                             figsize=(1800,2000), color_map=px.colors.sequential.Blues):
    layout2 = go.Layout(\
        title = title,\
        font = dict(family='SimHei, DengXian, "Microsoft YaHei"', size=24),\
        legend = {'traceorder':'normal'},\
        autosize = False, width=figsize[0], height=figsize[1],\
        margin = dict( l=50, r=50, b=100, t=100,pad=4),\
        )
    # fig1
    z = data_input.to_numpy()
    z_text = data_input.applymap(lambda x:"{x:0.3f}".format(x=x)).applymap(lambda x:x.replace("0.000","")).to_numpy()
    x=list(data_input.columns)
    y=list(data_input.index)
    
    # fig2
    fig2 = data_input.iplot(kind='bar', barmode='stack', \
                            asFigure=True, orientation='h', layout=layout2)#
    filename=fn["柱状图htm"].format(suffix = suffix)
    offline.plot(fig2, filename=filename)  

# --------------------------------------------------------------------------------------

def single_pyLDAvis(N, fin_tmpl, fout_tmpl, mds):
    filename = fin_tmpl.format(n=N)
    #print (filename)
    model_data = LDAp.load_model_from_pkl(filename)
    vis_data = pyLDAvis.prepare(**model_data, mds=mds)
    # pyLDAvis 2D可视化降维演算法有['PCOA','TSNE','MMDS'] 三种可能
    # 文档来源https://pyldavis.readthedocs.io/en/latest/modules/API.html#pyLDAvis.prepared_data_to_html
    html_out = fout_tmpl.format(n=N, kind=mds)
    
    #d3, ldavis, ldavis_css的资源需要先下载好并放在对映的目录，相对於html_out目录
    pyLDAvis.save_html(vis_data, html_out,\
                  d3_url="js/d3.min.js", \
                  ldavis_url='js/ldavis.v1.0.0.js', \
                  ldavis_css_url='js/ldavis.v1.0.0.css')
    return (model_data, html_out)


def batch_pyLDAvis(minN, maxN, fin_tmpl, fout_tmpl, mds, silent=False):
    M_list = []
    for n in range(minN, maxN + 1):  
        M, _HTML_ = single_pyLDAvis(n, fin_tmpl, fout_tmpl, mds)
        if silent == False:
            print ("exporting {_HTML_}".format(_HTML_=_HTML_))
        else:
            M_list.append(M)
            return(M_list)



if __name__ == '__main__':
    print ("module LDA visulalize")