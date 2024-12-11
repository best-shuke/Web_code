import os
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# 页面标题
st.title("🔍 K-Means 聚类分析")

# 页面简介
st.markdown("""
    👋 欢迎使用 K-Means 聚类工具！\n
    在这里，您可以对数据进行 K-Means 聚类分析，我们将帮助您从数据中发现潜在的分组模式。
    主要功能：
    - 📤 上传您的数据文件（CSV 格式）
    - 🧮 选择用于聚类的数值列
    - 📊 选择聚类簇的数量，并可通过肘部法则优化簇数
    - 🎨 可视化聚类结果
    - 📝 查看每个簇的描述性统计
""")

# 上传数据文件功能
st.header("📤 上传数据文件")
uploaded_file = st.file_uploader("选择一个CSV文件", type=["csv"])

# 如果文件已上传
if uploaded_file is not None:
    # 读取数据文件
    df = pd.read_csv(uploaded_file)
    st.success(f"文件 '{uploaded_file.name}' 已成功上传！🎉")

    # 显示数据预览
    st.subheader("📋 数据预览")
    st.write(df.head())

    # 获取数值型的列供用户选择
    numeric_columns = df.select_dtypes(include=['float64', 'int']).columns.tolist()

    # 确保用户选择了数值列
    if not numeric_columns:
        st.error("上传的数据中没有可用于聚类的数值型列，请检查您的数据。")
    else:
        selected_columns = st.multiselect("选择用于聚类的列", numeric_columns)

        if selected_columns:
            # 选择簇数
            n_clusters = st.number_input("选择簇数", min_value=2, max_value=10, value=3, step=1)

            # 距离度量选择
            distance_metric = st.selectbox("选择距离度量", ("欧式距离", "曼哈顿距离"))


            def calculate_distance(data, metric):
                """
                计算数据的距离矩阵，根据选择的度量方式返回不同的结果
                """
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
                    # 数据预处理（标准化）
                    transformed_data = calculate_distance(df[selected_columns], distance_metric)
                    scaler = StandardScaler()
                    transformed_data = scaler.fit_transform(transformed_data)

                    # KMeans 聚类
                    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
                    df['Cluster'] = kmeans.fit_predict(transformed_data)

                    # 存储聚类结果
                    cluster_result = df[['Cluster'] + selected_columns]

                    # 存储聚类结果到 CSV 文件
                    cluster_result_file = "cluster_result.csv"
                    cluster_result.to_csv(cluster_result_file, index=False)
                    st.success(f"聚类结果已保存到 '{cluster_result_file}' 文件！📁")

                    # 显示聚类结果
                    st.subheader("📊 聚类结果")
                    st.dataframe(cluster_result)

                    # 聚类的描述性统计
                    st.subheader("📝 每个簇的描述性统计")
                    cluster_description = df.groupby('Cluster')[selected_columns].describe()
                    st.write(cluster_description)

                    # 可视化聚类结果
                    st.subheader("🎨 聚类结果可视化")
                    fig = px.scatter(df, x=selected_columns[0], y=selected_columns[1], color='Cluster',
                                     title="K-Means 聚类结果")
                    st.plotly_chart(fig)

                    # 质心可视化
                    st.subheader("⭐ 聚类中心位置")
                    cluster_centers = pd.DataFrame(kmeans.cluster_centers_, columns=selected_columns)
                    st.write("每个簇的质心位置：")
                    st.write(cluster_centers)

                    # 质心可视化图
                    fig = px.scatter(cluster_centers, x=selected_columns[0], y=selected_columns[1],
                                     title="聚类中心位置")
                    st.plotly_chart(fig)

                except Exception as e:
                    st.error(f"聚类时出错: {e}")

        else:
            st.warning("请至少选择一列用于聚类。")

else:
    st.warning("⚠️ 请上传一个CSV文件以开始聚类分析。")
