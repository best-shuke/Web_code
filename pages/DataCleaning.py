import pandas as pd
import streamlit as st
import os
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# 设置页面标题
st.title('🔧 数据清洗工具')

# 页面简介
st.markdown("""
    👋 欢迎使用 **数据清洗工具**！

    在这里，您可以轻松地清洗您的数据，包括：
    - 🧹 去除重复行
    - 🧑‍🔬 填充缺失值
    - 🗑️ 删除不需要的列
    - ⚖️ 处理异常值
    - 📊 标准化或归一化数值数据
    \n
    请点击下面的按钮上传您的数据文件，我们将开始处理您的数据吧！🚀
""")

# 上传文件功能
st.header("📤 上传您的数据文件")
uploaded_file = st.file_uploader("选择一个CSV文件", type=["csv"])

if uploaded_file is not None:
    # 读取上传的文件
    df = pd.read_csv(uploaded_file)

    # 提示文件上传成功
    st.success(f"文件 '{uploaded_file.name}' 已成功上传！🎉")

    # 数据清洗选择
    st.header('🧹 选择数据清洗操作')

    # 去除重复行
    remove_duplicates = st.checkbox("去除重复行", value=False)

    # 删除指定列
    columns_to_drop = st.multiselect("选择要删除的列", df.columns.tolist())

    # 填充缺失值
    fill_missing = st.checkbox("填充缺失值")
    fill_value = None

    # 异常值处理
    handle_outliers = st.checkbox("处理异常值")
    outlier_threshold = st.number_input("设定异常值处理的标准差倍数", min_value=0.0, max_value=5.0, value=3.0, step=0.1)

    # 标准化/归一化
    scale_data = st.checkbox("数据标准化/归一化")
    scaler_type = None
    if scale_data:
        scaler_type = st.selectbox("选择标准化/归一化方式", ["标准化", "归一化"])


    # 自动修复 object 类型列
    def fix_object_columns(df):
        for column in df.select_dtypes(include=['object']).columns:
            # 尝试将文本列转换为类别数据，节省内存
            df[column] = df[column].astype('category')
        return df


    # 数据清洗操作
    if remove_duplicates:
        df = df.drop_duplicates()
        st.write("🧹 去除重复行后，数据行数:", len(df))

    if columns_to_drop:
        df = df.drop(columns=columns_to_drop)
        st.write(f"🗑️ 删除了列: {', '.join(columns_to_drop)}")

    # 填充缺失值
    if fill_missing:
        fill_value = st.selectbox("选择填充值", options=["平均值", "中位数", "常数值"])
        for column in df.columns:
            if df[column].isnull().any():
                if fill_value == "常数值":
                    fill_constant = st.number_input(f"请输入填充值（列：{column}）", value=0)
                    df[column] = df[column].fillna(fill_constant)
                elif fill_value == "平均值":
                    df[column] = df[column].fillna(df[column].mean())
                elif fill_value == "中位数":
                    df[column] = df[column].fillna(df[column].median())
        st.write("✅ 填充缺失值后，数据预览:")

    # 异常值处理
    if handle_outliers:
        for column in df.select_dtypes(include=['float64', 'int64']).columns:
            mean = df[column].mean()
            std = df[column].std()
            threshold_low = mean - outlier_threshold * std
            threshold_high = mean + outlier_threshold * std
            df = df[(df[column] >= threshold_low) & (df[column] <= threshold_high)]
            st.write(f"⚠️ 去除了列 {column} 中的异常值，阈值范围: ({threshold_low}, {threshold_high})")
            st.write(f"处理后的 {column} 列统计信息：")
            st.write(df[column].describe())

    # 标准化/归一化
    if scale_data:
        numerical_columns = df.select_dtypes(include=['float64', 'int64']).columns
        if len(numerical_columns) > 0:
            if scaler_type == "标准化":
                scaler = StandardScaler()
                df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
                st.write("📊 数据已完成标准化处理")
            elif scaler_type == "归一化":
                scaler = MinMaxScaler()
                df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
                st.write("📊 数据已完成归一化处理")
        else:
            st.warning("没有数值列可进行标准化或归一化处理。")

    # 自动修复 object 类型列
    df = fix_object_columns(df)

    # 显示清洗后的数据
    st.subheader('🔍 清洗后的数据预览')
    st.write(df.head())

    # 保存清洗后的数据为 data.csv
    df.to_csv('data.csv', index=False)

    st.success("🎉 数据已清洗并保存为 data.csv")

else:
    st.warning("⚠️ 请上传一个CSV文件来开始数据清洗。")
