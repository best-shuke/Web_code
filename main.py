import pandas as pd
import streamlit as st

# 设置应用标题和引导
st.title('📊 数据分析应用')
st.markdown("""
    欢迎使用本应用，您可以上传一个 CSV 文件，并通过以下功能进行数据分析：
    - **上传 CSV 文件**：上传一个数据文件以开始分析。
    - **数据类型确认与修改**：机器将自动推断数据类型，您可以进行确认和修改。
    - **数据查询**：您可以通过模糊查询或精确查询对数据进行筛选。
""")

# 上传文件功能
st.header('📥 上传数据文件')
uploaded_file = st.file_uploader("选择一个 CSV 文件", type="csv")

if uploaded_file is not None:
    # 读取上传的 CSV 文件
    df = pd.read_csv(uploaded_file)

    # 自动推断数据类型
    df = df.infer_objects()

    # 数据预览
    st.subheader('🔍 数据预览')
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

    # 保存修改后的数据为 modified_data.csv
    output_file = 'modified_data.csv'
    df.to_csv(output_file, index=False)
    st.success(f"✅ 数据已保存为 **{output_file}**")

    # 数据查询功能
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
        query_value = st.text_input(f"请输入模糊查询的关键字（列：{query_column}）")
        if query_value:
            if query_column == "全部":
                # 对所有列进行模糊查询
                result = df[df.apply(lambda row: row.astype(str).str.contains(query_value, na=False).any(), axis=1)]
            else:
                result = df[df[query_column].str.contains(query_value, na=False)]
            st.subheader(f"查询结果：")
            st.write(result)

    elif query_option == "精确查询":
        query_value = st.text_input(f"请输入精确查询的值（列：{query_column}）")
        if query_value:
            if query_column == "全部":
                # 对所有列进行精确查询
                result = df[df.apply(lambda row: row.astype(str).eq(query_value).any(), axis=1)]
            else:
                result = df[df[query_column] == query_value]
            st.subheader(f"查询结果：")
            st.write(result)
