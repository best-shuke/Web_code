import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# 页面标题
st.title("📊 数据可视化工具")

# 页面简介
st.markdown("""
    🎉 欢迎使用可视化工具！在这里，您可以轻松地选择不同的图表类型，
    并根据您的数据生成丰富的可视化效果。\n
    功能：
    - 📤 上传您的数据文件（CSV格式）
    - 🖼️ 选择不同类型的图表（散点图、折线图、饼图、箱线图等）
    - 🎨 自定义颜色、轴标签等
    - 💾 下载生成的图表
""")

# 上传数据文件功能
st.header("📤 上传数据文件")
uploaded_file = st.file_uploader("选择一个CSV文件", type=["csv"])

# 如果文件已上传
if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.success(f"文件 '{uploaded_file.name}' 已成功上传！🎉")

    # 显示数据预览
    st.subheader("📋 数据预览")
    st.write(df.head())

    # 图表类型选择
    st.subheader("📈 选择图表类型")
    plot_type = st.selectbox("选择图表类型", ["散点图", "折线图", "饼图", "箱线图", "条形图", "热力图"])
    x_axis = st.selectbox("选择X轴", df.columns)
    y_axis = st.selectbox("选择Y轴", df.columns)
    hue = st.selectbox("颜色分类（可选）", [None] + list(df.columns))

    # 图表样式定制选项
    st.subheader("🎨 图表样式定制")
    if plot_type in ["散点图", "折线图"]:
        marker_color = st.color_picker("选择数据点颜色", "#00f0f0")
    else:
        marker_color = None

    # 确保 session_state 初始化
    if "output" not in st.session_state:
        st.session_state["output"] = []

    if st.button("生成图表"):
        if plot_type == "散点图":
            if hue is None:
                fig = px.scatter(df, x=x_axis, y=y_axis, color_discrete_sequence=[marker_color])
            else:
                fig = px.scatter(df, x=x_axis, y=y_axis, color=hue, color_discrete_sequence=["blue", "green", "red"])

        elif plot_type == "折线图":
            if hue is None:
                fig = px.line(df, x=x_axis, y=y_axis, line_shape='linear', line_dash_sequence=['solid'], color_discrete_sequence=[marker_color])
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

        elif plot_type == "条形图":
            if hue is None:
                fig = px.bar(df, x=x_axis, y=y_axis, color_discrete_sequence=["lightblue"])
            else:
                fig = px.bar(df, x=x_axis, y=y_axis, color=hue)

        elif plot_type == "热力图":
            if hue is None:
                st.warning("热力图需要选择颜色分类（hue）列")
            else:
                fig = px.density_heatmap(df, x=x_axis, y=y_axis, color_continuous_scale="Viridis")

        # 显示生成的图表
        st.plotly_chart(fig)

        # 可选：如果需要将图表保存到 session_state，可以这样做
        st.session_state["output"].append(("chart", fig))

        # 下载图表功能
        st.subheader("💾 下载图表")
        img_bytes = fig.to_image(format="png")
        st.download_button(
            label="下载图表图片",
            data=img_bytes,
            file_name="chart.png",
            mime="image/png"
        )

else:
    st.warning("⚠️ 请上传一个CSV文件以开始生成图表。")
