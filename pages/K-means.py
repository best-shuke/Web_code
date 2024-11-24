# pages/K-means.py
import pandas as pd
import streamlit as st
from sklearn.cluster import KMeans
import numpy as np
import os
import plotly.express as px  # 导入 plotly.express 用于绘图

# 判断是否存在数据文件
if not os.path.isfile('data.csv'):
    st.error("请先执行上一步骤")
else:
    df = pd.read_csv('data.csv')
    st.subheader("K-Means 聚类")

    # 获取数值型的列供用户选择
    numeric_columns = df.select_dtypes(include=['float64', 'int']).columns.tolist()
    selected_columns = st.multiselect("选择用于聚类的列", numeric_columns)
    n_clusters = st.number_input("选择簇数", min_value=2, max_value=10, value=3, step=1)

    # 距离度量选择
    distance_metric = st.selectbox("选择距离度量", ("欧式距离", "曼哈顿距离"))


    def calculate_distance(data, metric):
        if metric == "欧式距离":
            return data  # 欧式距离不用处理，直接返回数据
        elif metric == "曼哈顿距离":
            return np.abs(data - data.mean())  # 曼哈顿距离计算


    # 显示肘部图
    if st.button("显示肘部图") and selected_columns:
        sse = []
        transformed_data = calculate_distance(df[selected_columns], distance_metric)

        # 使用不同簇数进行聚类
        for k in range(1, 11):
            kmeans = KMeans(n_clusters=k, random_state=0)
            kmeans.fit(transformed_data)
            sse.append(kmeans.inertia_)

        # 绘制肘部法则图
        fig = px.line(x=range(1, 11), y=sse, labels={'x': '簇数', 'y': '簇内误差平方和 (SSE)'})
        fig.update_traces(mode='lines+markers')
        st.plotly_chart(fig)  # 使用 Streamlit 显示图表

    # 执行聚类并显示结果
    if st.button("执行聚类") and selected_columns:
        try:
            transformed_data = calculate_distance(df[selected_columns], distance_metric)
            kmeans = KMeans(n_clusters=n_clusters, random_state=0)
            df['Cluster'] = kmeans.fit_predict(transformed_data)

            # 存储聚类结果
            st.session_state["clustered_data"] = df[['Cluster'] + selected_columns].head()  # 存储前5行数据

            # 存储簇的描述性统计
            cluster_description = df.groupby('Cluster')[selected_columns].describe()
            st.session_state["cluster_description"] = cluster_description  # 存储描述性统计

            # 显示聚类结果
            st.write(df[['Cluster'] + selected_columns].head())
            st.write(cluster_description)

        except Exception as e:
            st.error(f"聚类时出错: {e}")
