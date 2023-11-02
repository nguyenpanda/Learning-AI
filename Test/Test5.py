import matplotlib.pyplot as plt
import time
import random

# Thiết lập seed cho generator ngẫu nhiên
random.seed(2)


# Tạo dữ liệu giả định cho mục đích minh họa
def generate_data(num_samples=100, num_features=1):
    X = [[random.random() for _ in range(num_features)] for _ in range(num_samples)]
    y = [2 * x[0] + random.random() - 0.5 for x in X]
    return X, y


# Hàm tính loss
def compute_loss(X, y, weights):
    loss = 0
    for i in range(len(X)):
        prediction = sum(X[i][j] * weights[j] for j in range(len(weights)))
        loss += (prediction - y[i]) ** 2
    return loss


# Hàm tính gradient
def compute_gradient(X, y, weights):
    gradient = [0] * len(weights)
    for i in range(len(X)):
        prediction = sum(X[i][j] * weights[j] for j in range(len(weights)))
        error = prediction - y[i]
        for j in range(len(weights)):
            gradient[j] += error * X[i][j]
    return gradient


# Hàm huấn luyện Linear Regression model
def train_linear_regression(X, y, learning_rate=0.01, epochs=1000):
    # Khởi tạo weights ban đầu
    weights = [0] * len(X[0])
    history = []  # Tạo danh sách để lưu trữ lịch sử loss
    weight_history = []  # Tạo danh sách để lưu trữ lịch sử trọng số
    
    for epoch in range(epochs):
        total_loss = 0
        total_gradient = [0] * len(weights)
        for i in range(len(X)):
            prediction = sum(X[i][j] * weights[j] for j in range(len(weights)))
            error = prediction - y[i]
            total_loss += error ** 2
            for j in range(len(weights)):
                total_gradient[j] += error * X[i][j]
        
        # Cập nhật weights
        for i in range(len(weights)):
            weights[i] -= learning_rate * total_gradient[i] / len(X)
        
        history.append(total_loss)  # Lưu giá trị loss vào danh sách lịch sử
        weight_history.append(weights[:])  # Lưu trọng số vào danh sách lịch sử
        
        # In loss và trọng số sau mỗi epoch
        if epoch % 10 == 0:
            print(f'Epoch {epoch}, Loss: {total_loss / len(X)}, Weights: {weights}')
    
    return weights, history, weight_history


if __name__ == "__main__":
    # Sử dụng hàm train_linear_regression để huấn luyện mô hình và đo thời gian chạy
    start_time = time.time()
    X, y = generate_data()
    trained_weights, history, w_history = train_linear_regression(X, y)
    end_time = time.time()
    
    # In thời gian chạy của quá trình huấn luyện
    print("Running time: {:.2f} seconds".format(end_time - start_time))
    
    # In trọng số đã học được
    print("Final weights:", trained_weights)
    
    # Vẽ đồ thị lịch sử loss
    plt.plot(range(0, len(history) * 10, 10), history, color='green', label='Lịch sử Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
    
    # Vẽ đồ thị dữ liệu thực tế và dự đoán
    plt.scatter([x[0] for x in X], y, color='blue', label='Dữ liệu Thực tế')
    predicted_values = [sum(x[i] * trained_weights[i] for i in range(len(trained_weights))) for x in X]
    plt.plot([x[0] for x in X], predicted_values, color='red', linewidth=2, label='Dữ liệu Dự đoán')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()