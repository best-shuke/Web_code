import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA



# 设置应用标题和引导
st.title('📊 数据分析应用')
st.markdown("""
    欢迎使用本应用，您可以上传一个 CSV 文件，并通过以下模块进行数据分析：
    - **上传 CSV 文件**：上传一个数据文件以开始分析
    - **数据类型确认与修改**：机器将自动推断数据类型，您可以进行确认和修改
    - **数据查询**：您可以通过模糊查询或精确查询对数据进行筛选
    - **数据预处理**：处理缺失值、重复值，或是转换数据(如标准化)
    - **数据可视化**：进行数据探索或展示结果
    - **无监督学习**：提供K-means聚类分析和主成分分析(PCA) 
    - **线性回归**：提供单/多元线性模型和二分类Logistic回归模型
""")


#侧边栏选择模块
st.sidebar.title('📥 上传数据文件')  
uploaded_file = st.sidebar.file_uploader("上传CSV文件", type=["csv", "xlsx", "json"]) 
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file) 

function_choice = st.sidebar.selectbox("选择模块", ["数据类型确认与修改", "数据查询", "数据预处理", "数据可视化","无监督学习", "线性模型"])


 #数据类型确认与修改
if function_choice == "数据类型确认与修改":
        if uploaded_file is not None:
            # 自动推断数据类型
            df = df.infer_objects()

            # 数据预览
            st.subheader('👀 数据预览')
            st.write("这里是上传的数据预览：")
            st.write(df.head(10))

            # 数据整体描述
            st.subheader('📋 数据概述')
            st.write("### 数据描述统计")
            st.write(df.describe())  # 显示数值列的描述性统计
            st.write("### 数据类型与缺失值信息")
            st.write(df.info())  # 显示数据类型和缺失值

            # 显示推断的数据类型    
            st.subheader('📑 推断的数据类型')
            st.write("自动推断的数据类型如下：")
            st.dataframe(df.dtypes.to_frame().style.background_gradient(axis=0, cmap='coolwarm'))

            # 数据类型确认和修改
            st.header('🔧 数据类型确认与修改')
            st.markdown("""
                请查看每一列的数据类型。如果机器的推断不准确，您可以手动修改数据类型。
                例如，如果某列应该是日期，但被识别为字符串，您可以选择将其转换为时间数据类型。
            """)

            for column in df.columns:
                # 获取自动推断的类型
                inferred_type = df[column].dtype

                # 提供默认的选项，基于推断类型
                if inferred_type == 'object':
                    dtype_options = ["整数", "浮点数", "字符串", "时间数据"]
                    default_index = 2  # 默认为字符串
                elif inferred_type == 'int64':
                    dtype_options = ["整数", "浮点数", "字符串", "时间数据"]
                    default_index = 0  # 默认为整数
                elif inferred_type == 'float64':
                    dtype_options = ["整数", "浮点数", "字符串", "时间数据"]
                    default_index = 1  # 默认为浮点数
                elif inferred_type == 'datetime64[ns]':
                    dtype_options = ["整数", "浮点数", "字符串", "时间数据"]
                    default_index = 3  # 默认为时间数据
                else:
                    dtype_options = ["整数", "浮点数", "字符串", "时间数据"]
                    default_index = 2  # 默认为字符串

                # 让用户选择数据类型
                dtype = st.selectbox(f"选择列 **{column}** 的数据类型", options=dtype_options, index=default_index)

                # 根据用户选择的数据类型转换
                if dtype == "整数":
                    try:
                        df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0).astype(int)
                    except Exception as e:
                        st.error(f"列 {column} 转换为整数时出错: {e}")
                elif dtype == "浮点数":
                    try:
                        df[column] = pd.to_numeric(df[column], errors='coerce').fillna(0.0).astype(float)
                    except Exception as e:
                        st.error(f"列 {column} 转换为浮点数时出错: {e}")
                elif dtype == "时间数据":
                    try:
                        df[column] = pd.to_datetime(df[column], errors='coerce')
                    except Exception as e:
                        st.error(f"列 {column} 转换为时间数据时出错: {e}")
                else:
                    df[column] = df[column].astype(str)

            # 显示修改后的数据类型
            st.subheader('📝 修改后的数据类型')
            st.write("修改后的数据类型如下：")
            st.dataframe(df.dtypes.to_frame().style.background_gradient(axis=0, cmap='viridis'))

            # 显示修改后的数据预览
            st.subheader('🔄 修改后的数据预览')
            col1, col2 = st.columns([3, 1])  # 使用 3:1 的比例
            with col1:
                st.write(df.head(10))
            with col2:
                st.write(f"数据的总行数: {len(df)}")

            if st.button("保存数据"):
                # 保存修改后的数据为 modified_data.csv
                output_file = 'modified_data.csv'
                df.to_csv(output_file, index=False)
                st.success(f"✅ 数据已保存为 **{output_file}**")

#数据查询
elif function_choice == "数据查询":
    
    st.header('🔍 数据查询')
    st.markdown("""
        你可以通过以下方式对数据进行查询：
        - **模糊查询**：通过输入部分关键字进行搜索。
        - **精确查询**：查询完全匹配的值。
    """)

    query_option = st.selectbox("选择查询方式", ["模糊查询", "精确查询"])

    # 增加一个"全部"选项用于查询所有列
    query_column = st.selectbox("选择查询的列", options=["全部"] + list(df.columns))

    if query_option == "模糊查询":
        query_value = st.text_input(f"请输入模糊查询的关键字（列：{query_column})")
        if query_value:
            if query_column == "全部":
                # 对所有列进行模糊查询
                result = df[df.apply(lambda row: row.astype(str).str.contains(query_value, na=False).any(), axis=1)]
            else:
                result = df[df[query_column].str.contains(query_value, na=False)]
            st.subheader(f"查询结果：")
            st.write(result)

    elif query_option == "精确查询":
        query_value = st.text_input(f"请输入精确查询的值（列：{query_column})")
        if query_value:
            if query_column == "全部":
                # 对所有列进行精确查询
                result = df[df.apply(lambda row: row.astype(str).eq(query_value).any(), axis=1)]
            else:
                result = df[df[query_column] == query_value]
            st.subheader(f"查询结果：")
            st.write(result)

#数据预处理
elif function_choice == "数据预处理":
    st.header('🛀 数据预处理')
    st.markdown("""
        对数据预处理可以提高数据质量，通常有以下几种方式：
        - 1. 删除缺失值行 
        - 2. 填充缺失值行：自定义填充
        - 3. 删除重复行
        - 4. 标准化数据：使其符合均值为0, 方差为1的分布
    """)

    # 数据预览
    st.subheader('数据预览')
    st.write(df.head(10))

    #选择清理方式    
    clean_option = st.selectbox("清洗选项", ["删除缺失值行", "填充缺失值", "删除重复行", "标准化数据"])
    
    if clean_option == "删除缺失值行" and st.button("执行操作"):
        df = df.dropna()
        st.success("已删除所有包含缺失值的行。")
        st.write(df.head())

    elif clean_option == "填充缺失值":
        fill_column = st.selectbox("选择要填充缺失值的列", options=df.columns)
        fill_value = st.text_input("填充值", "")
        if st.button("执行操作") and fill_value:
            df[fill_column] = df[fill_column].fillna(fill_value)
            st.success(f"已将列 `{fill_column}` 的缺失值填充为 `{fill_value}`。")
            st.write(df.head())

    elif clean_option == "删除重复行" and st.button("执行操作"):
        df = df.drop_duplicates()
        st.success("已删除重复行")
        st.write(df.head())

    elif clean_option == "标准化数据":
        numeric_columns = df.select_dtypes(include=['float64']).columns.tolist()
        selected_columns = st.multiselect("选择需要标准化的列", numeric_columns)
        df[selected_columns] = df[selected_columns].apply(lambda x:(x-x.mean())/x.std())
        if selected_columns is not None:
            st.subheader('📝 预览标准化后的数据')
            st.write(df.head())

    if st.button("保存数据"):
        # 保存修改后的数据为 modified_data.csv
        output_file = 'modified_data.csv'
        df.to_csv(output_file, index=False)
        st.success(f"✅ 数据已保存为 **{output_file}**")


#数据可视化
elif function_choice == "数据可视化":
    st.header('📊 数据可视化')
    st.markdown("""
        不同的图表类型帮助我们进行数据探索和结果展示：
        - **散点图**：适合展示两个连续变量之间的关系
        - **折线图**：适合展示数据随时间变化的趋势
        - **饼图**：适合展示各部分占总体的比例关系，但当类别过多时，饼图会变得难以阅读
        - **箱线图**：箱线图展示了数据的中位数、四分位数和异常值，可以直观地看出数据的分布和离散程度；适用于比较不同组数据的分布情况，识别异常值。
    """)
    # 数据预览
    st.subheader('数据预览')
    st.write(df.head(10))

    plot_type = st.selectbox("选择图表类型", ["散点图", "折线图", "饼图", "箱线图"])
    x_axis = st.selectbox("选择X轴", df.columns)
    y_axis = st.selectbox("选择Y轴", df.columns)
    hue = st.selectbox("颜色分类（可选）", [None] + list(df.columns))

    st.header('📃生成图表')
    if st.button("生成图表"):
        if plot_type == "散点图":
            fig = px.scatter(df, x=x_axis, y=y_axis, color=hue)
        elif plot_type == "折线图":
            fig = px.line(df, x=x_axis, y=y_axis, color=hue)
        elif plot_type == "饼图":
            fig = px.pie(df, names=hue, title=f"{hue} 分布")
        elif plot_type == "箱线图":
            fig = px.box(df, x=hue, y=y_axis)

        st.write(fig)
    
        #保存图表
        if st.button("保存图表"):
            output_figure = 'output_figure.png'
            fig.write_image('output_figure.png')
            st.success(f"✅ 图表已保存为 **{output_figure}**")

elif function_choice == "无监督学习":
    st.header('🍪 无监督学习')
    st.markdown("""
        无监督学习是机器学习中的一种方法，与监督学习相对。
        在无监督学习中，训练数据只包含输入数据而不包含标签，目的是从数据中发现模式或结构，而不是对数据进行预测。
        - **K-means聚类分析**：
            K-means聚类的核心思想是将数据点划分为K个簇，使得簇内的数据点尽可能相似，而簇间的数据点尽可能不同。
            这个过程通过迭代地移动簇中心和重新分配数据点到最近的簇中心来实现，直到簇的分配不再发生变化或达到预设的迭代次数
        - **主成分分析(PCA)：
            主成分分析（PCA）是一种降维技术，它通过正交变换将一组可能相关的变量转换成一组线性不相关的变量，称为主成分。
            其核心思想是识别数据中的主要变化方向，并在这些方向上捕捉数据的大部分信息，从而用较少的维度来表示原始数据集。
    """)
    # 数据预览
    st.subheader('数据预览')
    st.write(df.head(10))

    learning_option = st.selectbox("选择模型", ["K-means聚类分析", "主成分分析(PCA)"])

    if learning_option == "K-means聚类分析":
        st.subheader("K-means聚类分析")
    
        # 获取数值型的列供用户选择
        numeric_columns = df.select_dtypes(include=['float64', 'int']).columns.tolist()
        selected_columns = st.multiselect("选择用于聚类的列", numeric_columns)
        n_clusters = st.number_input("选择簇数", min_value=2, max_value=10, value=3, step=1)

        # 距离度量选择
        distance_metric = st.selectbox("选择距离度量", ("欧式距离", "曼哈顿距离"))

        def calculate_distance(data, metric):
            if metric == "欧式距离":
                return data
            elif metric == "曼哈顿距离":
                return np.abs(data - data.mean())

        if st.button("显示肘部图") and selected_columns:
            sse = []
            transformed_data = calculate_distance(df[selected_columns], distance_metric)
            for k in range(1, 11):
                kmeans = KMeans(n_clusters=k, random_state=0)
                kmeans.fit(transformed_data)
                sse.append(kmeans.inertia_)

            fig = px.line(x=range(1, 11), y=sse, labels={'x': '簇数', 'y': '簇内误差平方和 (SSE)'})
            fig.update_traces(mode='lines+markers')
            st.write(fig)

        if st.button("执行聚类") and selected_columns:
            try:
                transformed_data = calculate_distance(df[selected_columns], distance_metric)
                kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                df['Cluster'] = kmeans.fit_predict(transformed_data)

                # 存储聚类结果
                st.write(df[['Cluster'] + selected_columns].head())

                # 存储簇的描述性统计
                cluster_description = df.groupby('Cluster')[selected_columns].describe()
                st.write(cluster_description)


            except Exception as e:
                st.error(f"聚类时出错: {e}")

    elif learning_option == "主成分分析":
        st.subheader("主成分分析(PCA)")

        # 选择数值列用于主成分分析
        numeric_columns = df.select_dtypes(include=['float64', 'int']).columns.tolist()
        selected_columns = st.multiselect("选择用于主成分分析的列", numeric_columns)

        if selected_columns:
            n_components = st.number_input("选择主成分数量", min_value=1, max_value=len(selected_columns), value=2, step=1)
            if st.button("执行PCA"):
                try:
                    pca = PCA(n_components=n_components)
                    pca_result = pca.fit_transform(df[selected_columns])

                    # 保存PCA结果
                    pca_df = pd.DataFrame(pca_result, columns=[f"PC{i + 1}" for i in range(n_components)])
                    st.write(pca_df.head())

                    # 显示方差贡献率
                    explained_variance = pca.explained_variance_ratio_ * 100
                    variance_df = pd.DataFrame(
                        {"主成分": [f"PC{i + 1}" for i in range(n_components)], "方差贡献率 (%)": explained_variance})
                    st.write("各主成分的方差贡献率：")
                    st.write(variance_df)

                    # 如果有两个或更多主成分，绘制散点图
                    if n_components >= 2:
                        fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA 散点图 (前两个主成分)", labels={"PC1": "主成分 1", "PC2": "主成分 2"})
                        st.write(fig)
                except Exception as e:
                    st.error(f"主成分分析时出错: {e}")
        else:
            st.info("请选择至少一列用于主成分分析")

elif function_choice == "线性模型":
    st.header('📈 线性模型')
    st.markdown("""
        在本模块，我们提供单/多元线性回归模型和二分类逻辑回归模型：
        - **单/多元线性回归**：
            线性回归是一种统计学方法，用于建立一个或多个自变量与连续型因变量之间的线性关系，
            通过最小二乘法最小化观测值和预测值之间的差异。
        - **二分类Logistic回归**：
            二分类Logistic回归使用因变量为二分类(如成功/失败，是/否)的数据通过训练数据来估计模型参数，
            并使用这些参数来预测新数据的分类结果。
        """)

    # 数据预览
    st.subheader('数据预览')
    st.write(df.head(10))
    
    model_option = st.selectbox("选择模型", ["单/多元线性回归", "二分类Logistic回归"])
 
    #单/多元线性回归
    if model_option == "单/多元线性回归":
        st.subheader("单/多元线性回归")
        from sklearn.linear_model import LinearRegression

        columns = df.select_dtypes(include=['float64', 'int']).columns.tolist()
        dependent_variable = st.multiselect("选择因变量", columns)
        if len(dependent_variable)!=1:
            st.info("请选择一列作为因变量")
        independent_variable = st.multiselect("选择自变量", columns)
        y=df[dependent_variable].astype('float')
        x=df[independent_variable].astype('float')


        if independent_variable:
            #模型拟合
            if st.button("执行单/多元线性回归"):
                try:
                    import statsmodels.api as sm
                    X=sm.add_constant(x)#添加常数列，即截距项
                    model=sm.OLS(y,X)#创建线性回归模型
                    result=model.fit()#拟合模型
                    params=result.params#系数矩阵
                    st.write(result.summary())#查看回归结果
                except Exception as e:
                    st.error(f"单/多元线性回归时出错: {e}")  

    #二分类逻辑回归
    elif model_option == "二分类Logistic回归":
        st.subheader("二分类Logistic回归")

        import statsmodels.formula.api as smf
        import statsmodels.api as sm
        from scipy.stats import norm
        import numpy as np

        # 选择用于回归的列
        columns = df.select_dtypes(include=['float64', 'int', 'string']).columns.tolist()
        dependent_variable = st.multiselect("选择因变量", columns)
        if len(dependent_variable)!=1:
            st.info("请选择一列作为因变量")
        independent_variable = st.multiselect("选择自变量", columns)
        df[dependent_variable]=df[dependent_variable].astype('float')
        df[independent_variable]=df[independent_variable].astype('float')
        x = '+'.join(independent_variable)
        y = dependent_variable[0]

        if independent_variable:

            #模型拟合
            if st.button("执行二分类Logistic回归"):
                try:
                    def logistic_regression(y,x,df):
                        model = smf.glm(formula = f'{y} ~ {x}',
                                        data = df,
                                        family=sm.families.Binomial()).fit()  
                        results_as_html = model.summary().tables[1].as_html()
                        result = pd.read_html(results_as_html, header=0, index_col=0)[0]
                        st.write(result)
                    logistic_regression(y=y,x=x,df=df)
                except Exception as e:
                    st.error(f"Logistic回归时出错: {e}")

            #计算OR值
            if st.button("计算OR值"):
                try:
                    def OR(y,x,df):
                        model = smf.glm(formula = f'{y} ~ {x}',
                                        data = df,
                                        family=sm.families.Binomial()).fit()  
                        stat = pd.DataFrame({'p': model.pvalues,                      
                                             'OR': np.exp(model.params), 
                                            'OR_lower_ci': np.exp(model.params - norm.ppf(0.975)*model.bse),
                                            'OR_upper_ci': np.exp(model.params + norm.ppf(0.975)*model.bse)}) 
                        stat['sig'] = stat.apply(lambda x : "*" if x['p']<0.05 else "no_sig",axis=1)
                        stat= stat.sort_values('OR', ascending=True)                            
                        st.write(stat)
                    OR(y=y,x=x,df=df)
                except Exception as e:
                    st.error(f"计算OR值时出错: {e}")

            #绘制OR森林图
            if st.button("绘制OR森林图"):
                try:
                    def OR_plot(y,x,df):
                        model = smf.glm(formula = f'{y} ~ {x}',
                                        data = df,
                                        family=sm.families.Binomial()).fit()  
                        stat = pd.DataFrame({'p': model.pvalues,                      
                                             'OR': np.exp(model.params), 
                                            'OR_lower_ci': np.exp(model.params - norm.ppf(0.975)*model.bse),
                                            'OR_upper_ci': np.exp(model.params + norm.ppf(0.975)*model.bse)}) 
                        stat['sig'] = stat.apply(lambda x : "*" if x['p']<0.05 else "no_sig",axis=1)
                        stat= stat.sort_values('OR', ascending=True)                            
                        forest_df = stat.drop("Intercept")\
                                        .reset_index()\
                                        .rename(columns={'index': 'independent_var'})\
                                        .sort_values('OR', ascending=False)
                        from plotnine import ggplot,aes,geom_point,geom_errorbarh,scale_color_manual,scale_y_discrete,guides,guide_legend,labs,geom_vline,theme_minimal,theme,element_text
                        forest = ggplot(forest_df , 
                                        aes(y='independent_var', x='OR')) + geom_point(aes(color='sig'),size=2) + geom_errorbarh(aes(xmin='OR_lower_ci', xmax='OR_upper_ci',color ='sig'), height=0.1) + scale_color_manual(values = ["red","black"]) + scale_y_discrete(limits= forest_df["independent_var"]) + guides(color=guide_legend(reverse=True))+labs(title='logistic Regression', x='OR', y='variable')+geom_vline(xintercept=1, linetype='dashed', color='black')+theme_minimal()+theme(plot_title=element_text(hjust=0.5))
                        st.pyplot(ggplot.draw(forest)) 
                    OR_plot(y=y,x=x,df=df)
                except Exception as e:
                    st.error(f"绘制OR森林图时出错: {e}")

        else:
            st.info("请选择至少一列作为自变量")        
        

