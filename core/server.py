import streamlit as st
import json
import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import os
from prediction import predict_stock_price
import matplotlib.dates as mdates

# Set Streamlit page layout to wide mode - Moved into main()
# st.set_page_config(layout="wide") # This will now be called inside main()

# Function to plot metrics
def plot_metrics(json_path):
    with open(json_path, 'r') as f:
        metrics = json.load(f)

    try:
        df = pd.read_csv('./datasets/test_data.csv')
    except FileNotFoundError:
        st.error("错误：未找到文件 './datasets/test_data.csv'。")
        return None

    # Reduce chart height
    fig, axes = plt.subplots(1, 4, figsize=(28, 5))

    # Plot Volume bar chart
    axes[0].bar(range(len(df['Volume'])), df['Volume'], color='r')
    axes[0].set_title('Volume', color='blue')

    axes[1].plot(metrics["epochs"], metrics["train_acc"], label="Train Accuracy", color='b')
    axes[1].plot(metrics["epochs"], metrics["val_acc"], label="Val Accuracy", color='r')
    axes[1].set_title('Accuracy', color='blue')
    axes[1].legend()

    axes[2].plot(metrics["epochs"], metrics["train_mse"], label="Train MSE", color='b')
    axes[2].plot(metrics["epochs"], metrics["val_mse"], label="Val MSE", color='r')
    axes[2].set_title('Mean Squared Error', color='blue')
    axes[2].legend()

    axes[3].plot(metrics["epochs"], metrics["train_rmse"], label="Train RMSE", color='b')
    axes[3].plot(metrics["epochs"], metrics["val_rmse"], label="Val RMSE", color='r')
    axes[3].set_title('Root Mean Squared Error', color='blue')
    axes[3].legend()

    # Set axis labels and tick colors to blue
    for ax in axes.flat:
        ax.set_xlabel(ax.get_xlabel(), color='blue')
        ax.set_ylabel(ax.get_ylabel(), color='blue')
        for tick in ax.get_xticklabels():
            tick.set_color('blue')
        for tick in ax.get_yticklabels():
            tick.set_color('blue')

    plt.subplots_adjust(left=0.03, right=0.97, top=0.9, bottom=0.1)  # Adjust subplot margins
    return fig

# Main function to run the Streamlit app
def main():
    # Set Streamlit page layout to wide mode
    st.set_page_config(layout="wide")

    # Create tabs
    tab1, tab2, tab3 = st.tabs(["LSTM", "Transformer", "预测"])

    # LSTM tab
    with tab1:
        col1, col2 = st.columns([1, 2])
        with col1:
            lstm_img_path = "./report/lstm_report_table_k.png"
            st.image(lstm_img_path, use_container_width=True)
        with col2:
            st.write("该 `LSTM` 模型是一个基于 PyTorch 构建的神经网络模型，主要由两部分构成。第一部分是一个 LSTM 层，它接收输入数据，输入数据的特征维度由 `input_size` 决定，隐藏层的维度由 `hidden_layer_size` 决定，并且设置了 `batch_first=True`，意味着输入数据的第一个维度代表批量大小。第二部分是一个全连接层（FC 层），它将 LSTM 层的输出映射到指定的输出维度，这个输出维度由 `output_size` 决定。在模型的前向传播过程中，输入数据首先经过 LSTM 层处理，然后取 LSTM 层输出序列的最后一个时间步的特征，再将其输入到全连接层，最终得到模型的输出。")
        lstm_json_path = "./report/lstm_train_report.json"
        lstm_fig = plot_metrics(lstm_json_path)
        if lstm_fig:
            st.pyplot(lstm_fig, bbox_inches='tight')

    # Transformer tab
    with tab2:
        col1, col2 = st.columns([1, 2])
        with col1:
            tf_img_path = "./report/tf_report_table_k.png"
            st.image(tf_img_path, use_container_width=True)
        with col2:
            st.markdown("该 `Transformer` 模型基于 PyTorch 构建，主要由以下部分构成：位置编码模块：`LearnedPositionalEncoding` 类，利用 `nn.Embedding` 学习位置嵌入，接收 `max_len` 和 `d_model` 参数，为输入添加位置信息。嵌入层：`nn.Linear` 层将输入从 `input_size` 映射到 `d_model` 维度。自注意力层：两个 `nn.MultiheadAttention` 层，参数为 `d_model`、`nhead` 和 `dropout`，`batch_first=True`，用于捕捉序列元素关系。全连接层：`fc1` 层经 `ReLU` 激活，`fc_out` 层将输出映射到 `output_size` 维度。丢弃层：使用 `nn.Dropout` 防止过拟合。前向传播时，输入先嵌入并添加位置编码，经两次自注意力层及残差连接，取最后时间步特征，再经全连接层得到最终输出。")
        tf_json_path = "./report/tf_train_report.json"
        tf_fig = plot_metrics(tf_json_path)
        if tf_fig:
            st.pyplot(tf_fig, bbox_inches='tight')

    # Prediction tab
    with tab3:
        uploaded_file = st.file_uploader("上传 CSV 文件", type=["csv"])
        if uploaded_file is not None:
            # Save the uploaded file locally
            with open(os.path.join(".", uploaded_file.name), "wb") as f:
                f.write(uploaded_file.getbuffer())
            price = predict_stock_price(os.path.join(".", uploaded_file.name))
            if price is not None:
                st.success(f"预测未来一天的价格: {price:.4f}")

                # Read the CSV file
                df = pd.read_csv(os.path.join(".", uploaded_file.name))

                # Convert Date column to datetime type
                df['Date'] = pd.to_datetime(df['Date'])

                # Create chart
                fig, ax1 = plt.subplots(figsize=(12, 2))

                # Create a second y-axis to display Close price
                ax2 = ax1.twinx()

                # Plot historical Close price (blue dots and line)
                ax2.plot(df['Date'], df['Close'], 'bo-', label='Historical Close', markersize=4)

                # Get the last date and calculate the next day's date
                last_date = df['Date'].iloc[-1]
                next_date = last_date + pd.Timedelta(days=1)

                # Plot predicted price (red dot)
                ax2.plot(next_date, price, 'ro', label='Predicted Price')

                # Draw a red line from the last historical point to the predicted point
                if len(df) > 0:
                    last_close = df['Close'].iloc[-1]
                    ax2.plot([last_date, next_date], [last_close, price], 'r-')

                ax2.set_ylabel('Price', color='blue')
                ax2.tick_params(axis='y', labelcolor='blue')

                # Set x-axis date format, only display month and day
                ax2.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d'))
                ax2.xaxis.set_major_locator(mdates.DayLocator())

                # Add legend
                lines, labels = ax1.get_legend_handles_labels()
                lines2, labels2 = ax2.get_legend_handles_labels()
                ax2.legend(lines + lines2, labels + labels2, loc='upper left')

                # Set title
                plt.title('Historical Close Price and Predicted Price')

                # Display chart
                st.pyplot(fig)

# This ensures main() is called only when the script is executed directly by Python
if __name__ == "__main__":
    main()