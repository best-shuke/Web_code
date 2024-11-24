import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.decomposition import PCA

st.header("主成分分析 (PCA)")

# 判断是否存在数据文件
if not os.path.isfile('data.csv'):
    st.error("请先执行上一步骤")
else:
    df = pd.read_csv('data.csv')

    # 选择数值列用于主成分分析
    numeric_columns = df.select_dtypes(include=['float64', 'int']).columns.tolist()
    selected_columns = st.multiselect("选择用于主成分分析的列", numeric_columns)

    if selected_columns:
        # 执行PCA以计算所有主成分的方差贡献率
        pca = PCA()
        pca_result = pca.fit_transform(df[selected_columns])

        # 获取方差贡献率
        explained_variance = pca.explained_variance_ratio_ * 100

        # 绘制碎石图 (Scree Plot)
        fig_scree = go.Figure()
        fig_scree.add_trace(go.Scatter(
            x=[f"PC{i + 1}" for i in range(len(explained_variance))],
            y=explained_variance,
            mode='markers+lines',
            name='方差贡献率',
            marker=dict(color='blue', size=10)
        ))
        fig_scree.update_layout(
            title="碎石图 (Scree Plot)",
            xaxis_title="主成分",
            yaxis_title="方差贡献率 (%)",
            showlegend=False
        )
        st.plotly_chart(fig_scree)

        # 用户选择主成分数量
        n_components = st.number_input("选择主成分数量", min_value=1, max_value=len(selected_columns), value=2, step=1)

        if st.button("执行PCA"):
            try:
                # 执行PCA并限制主成分数量
                pca = PCA(n_components=n_components)
                pca_result = pca.fit_transform(df[selected_columns])

                # 保存PCA结果
                pca_df = pd.DataFrame(pca_result, columns=[f"PC{i + 1}" for i in range(n_components)])
                st.write("PCA 结果：")
                st.write(pca_df.head())

                # 显示选择的主成分方差贡献率
                explained_variance_selected = pca.explained_variance_ratio_ * 100
                variance_df = pd.DataFrame(
                    {"主成分": [f"PC{i + 1}" for i in range(n_components)],
                     "方差贡献率 (%)": explained_variance_selected})
                st.write("各主成分的方差贡献率：")
                st.write(variance_df)

                # 如果有两个或更多主成分，绘制散点图
                if n_components >= 2:
                    fig = px.scatter(pca_df, x="PC1", y="PC2", title="PCA 散点图 (前两个主成分)",
                                     labels={"PC1": "主成分 1", "PC2": "主成分 2"})
                    st.plotly_chart(fig)  # 使用 Streamlit 显示图表

                # 如果主成分大于2，提供其他可视化方式
                if n_components > 2:
                    st.write("更多主成分的可视化可以通过其他方法实现。")
            except Exception as e:
                st.error(f"主成分分析时出错: {e}")
    else:
        st.info("请选择至少一列用于主成分分析")
