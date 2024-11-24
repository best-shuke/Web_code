import pandas as pd
import streamlit as st
import plotly.express as px
import os

# 判断是否存在数据文件
if not os.path.isfile('data.csv'):
    st.error("请先执行上一步骤")
else:
    df = pd.read_csv('data.csv')

    st.subheader("可视化")

    # 图表类型选择
    plot_type = st.selectbox("选择图表类型", ["散点图", "折线图", "饼图", "箱线图"])
    x_axis = st.selectbox("选择X轴", df.columns)
    y_axis = st.selectbox("选择Y轴", df.columns)
    hue = st.selectbox("颜色分类（可选）", [None] + list(df.columns))

    # 确保 session_state 初始化
    if "output" not in st.session_state:
        st.session_state["output"] = []

    if st.button("生成图表"):
        if plot_type == "散点图":
            if hue is None:
                fig = px.scatter(df, x=x_axis, y=y_axis)  # 如果没有颜色分类，则不使用 color 参数
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis, color=hue)

        elif plot_type == "折线图":
            if hue is None:
                fig = px.line(df, x=x_axis, y=y_axis)  # 如果没有颜色分类，则不使用 color 参数
            else:
                fig = px.line(df, x=x_axis, y=y_axis, color=hue)

        elif plot_type == "饼图":
            if hue is None:
                st.warning("饼图需要选择颜色分类（hue）列")
            else:
                fig = px.pie(df, names=hue, title=f"{hue} 分布")

        elif plot_type == "箱线图":
            if hue is None:
                fig = px.box(df, y=y_axis)  # 如果没有颜色分类，则不使用 x 参数
            else:
                fig = px.box(df, x=hue, y=y_axis)

        # 显示生成的图表
        st.plotly_chart(fig)

        # 可选：如果需要将图表保存到 session_state，可以这样做
        st.session_state["output"].append(("chart", fig))
