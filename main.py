import pandas as pd
import streamlit as st

# 设置应用标题
st.title('数据分析应用')

# 上传文件功能
st.header('上传数据文件')
uploaded_file = st.file_uploader("选择一个 CSV 文件", type="csv")

if uploaded_file is not None:
    # 读取上传的 CSV 文件
    df = pd.read_csv(uploaded_file)

    # 显示数据的前几行预览
    st.subheader('数据预览')
    st.write(df.head(10))

    # 数据清洗选择
    st.header('选择数据清洗操作')

    # 去除重复行
    remove_duplicates = st.checkbox("去除重复行", value=False)

    # 处理数据清洗
    if remove_duplicates:
        df = df.drop_duplicates()
        st.write("去除重复行后，数据行数:", len(df))

    flag = 0

    # 填充缺失值
    fill_missing = st.checkbox("填充缺失值")
    fill_value = None
    if fill_missing:
        fill_value = st.selectbox("选择填充值", options=["平均值", "中位数", "常数值"],)
        for column in df.columns:
            if df[column].isnull().any():
                if fill_value == "常数值":
                    fill_constant = st.number_input(f"请输入填充值（列：{column}）", value=0)
                    df[column] = df[column].fillna(fill_constant)
                elif fill_value == "平均值":
                    df[column] = df[column].fillna(df[column].mean())
                elif fill_value == "中位数":
                    df[column] = df[column].fillna(df[column].median())

        flag = 1

        st.write("填充缺失值后，数据预览:")

    # 显示清洗后的数据
    st.subheader('清洗后的数据预览')
    if flag==1:
        st.write(df.head())

        # 保存清洗后的数据为 data.csv
        df.to_csv('data.csv', index=False)

        st.success("数据已清洗并保存为 data.csv")
