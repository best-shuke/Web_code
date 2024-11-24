import os

import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
import streamlit as st
from scipy.stats import norm

# 判断是否存在数据文件
if not os.path.isfile('data.csv'):
    st.error("请先执行上一步骤")
else:
    df = pd.read_csv('data.csv')

    st.subheader("二分类Logistic回归")

    # 选择用于回归的列
    columns = df.select_dtypes(include=['float64', 'int', 'string']).columns.tolist()
    dependent_variable = st.multiselect("选择因变量", columns)
    independent_variable = st.multiselect("选择自变量", columns)

    if len(dependent_variable) != 1:
        st.info("请选择一列作为因变量")

    if independent_variable:
        # 确保数据转换为 float
        df[dependent_variable] = df[dependent_variable].astype('float')
        df[independent_variable] = df[independent_variable].astype('float')

        x = '+'.join(independent_variable)
        y = dependent_variable[0]

        # 模型拟合函数
        def logistic_regression(y, x, df):
            model = smf.glm(formula=f'{y} ~ {x}',
                            data=df,
                            family=sm.families.Binomial()).fit()
            results_as_html = model.summary().tables[1].as_html()
            result = pd.read_html(results_as_html, header=0, index_col=0)[0]
            return result


        # 计算OR值函数
        def calculate_OR(y, x, df):
            model = smf.glm(formula=f'{y} ~ {x}',
                            data=df,
                            family=sm.families.Binomial()).fit()
            stat = pd.DataFrame({'p': model.pvalues,
                                 'OR': np.exp(model.params),
                                 'OR_lower_ci': np.exp(model.params - norm.ppf(0.975) * model.bse),
                                 'OR_upper_ci': np.exp(model.params + norm.ppf(0.975) * model.bse)})
            stat['sig'] = stat.apply(lambda x: "*" if x['p'] < 0.05 else "no_sig", axis=1)
            stat = stat.sort_values('OR', ascending=True)
            return stat


        # 绘制OR森林图函数
        def OR_forest_plot(y, x, df):
            model = smf.glm(formula=f'{y} ~ {x}',
                            data=df,
                            family=sm.families.Binomial()).fit()
            stat = pd.DataFrame({'p': model.pvalues,
                                 'OR': np.exp(model.params),
                                 'OR_lower_ci': np.exp(model.params - norm.ppf(0.975) * model.bse),
                                 'OR_upper_ci': np.exp(model.params + norm.ppf(0.975) * model.bse)})
            stat['sig'] = stat.apply(lambda x: "*" if x['p'] < 0.05 else "no_sig", axis=1)
            stat = stat.sort_values('OR', ascending=True)
            forest_df = stat.drop("Intercept") \
                .reset_index() \
                .rename(columns={'index': 'independent_var'}) \
                .sort_values('OR', ascending=False)

            from plotnine import ggplot, aes, geom_point, geom_errorbarh, scale_color_manual, scale_y_discrete, \
                guides, guide_legend, labs, geom_vline, theme_minimal, theme, element_text

            forest = ggplot(forest_df,
                            aes(y='independent_var', x='OR')) + geom_point(aes(color='sig'),
                                                                           size=2) + geom_errorbarh(
                aes(xmin='OR_lower_ci', xmax='OR_upper_ci', color='sig'), height=0.1) + scale_color_manual(
                values=["red", "black"]) + scale_y_discrete(limits=forest_df["independent_var"]) + guides(
                color=guide_legend(reverse=True)) + labs(title='Logistic Regression OR Forest Plot', x='OR',
                                                         y='Variable') + geom_vline(xintercept=1, linetype='dashed',
                                                                                    color='black') + theme_minimal() + theme(
                plot_title=element_text(hjust=0.5))

            return forest


        # 执行Logistic回归
        if st.button("执行二分类Logistic回归"):
            try:
                result = logistic_regression(y=y, x=x, df=df)
                st.write("回归结果：")
                st.write(result)
            except Exception as e:
                st.error(f"Logistic回归时出错: {e}")

        # 计算OR值
        if st.button("计算OR值"):
            try:
                stat = calculate_OR(y=y, x=x, df=df)
                st.write("OR值和置信区间：")
                st.write(stat)
            except Exception as e:
                st.error(f"计算OR值时出错: {e}")

        # 绘制OR森林图
        if st.button("绘制OR森林图"):
            try:
                forest = OR_forest_plot(y=y, x=x, df=df)
                st.write("OR森林图：")
                st.pyplot(forest)
            except Exception as e:
                st.error(f"绘制OR森林图时出错: {e}")

    else:
        st.info("请选择至少一列作为自变量")
