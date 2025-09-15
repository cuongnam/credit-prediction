# 🔮 Dự đoán rủi ro tín dụng bằng XGBoost

## 📌 Bài làm về gì
Đây là một project học máy nhằm **dự đoán khả năng rủi ro tín dụng của khách hàng** dựa trên dữ liệu trong bộ `Give Me Some Credit` (Kaggle).  
Mục tiêu chính: xây dựng một mô hình dự đoán giúp ngân hàng/tổ chức tài chính đánh giá **khách hàng có khả năng vỡ nợ hay không** dựa trên các thông tin cơ bản (tuổi, thu nhập, số lần trễ hạn, tỷ lệ tín dụng quay vòng, số người phụ thuộc, ...).  
Kết quả dự đoán:
- **0** → Khả năng tín dụng tốt (ít rủi ro)  
- **1** → Rủi ro cao  

---

## 🛠️ Sử dụng công nghệ, thuật toán, ngôn ngữ lập trình gì
- **Ngôn ngữ**: Python 3.10+  
- **Machine Learning**: [XGBoost](https://xgboost.readthedocs.io/) – một thuật toán Gradient Boosting mạnh mẽ, tối ưu cho các bài toán phân loại và dự đoán rủi ro.  
- **Thư viện**:
  - `xgboost`, `scikit-learn`, `numpy`, `pandas` để huấn luyện và đánh giá mô hình.  
  - `matplotlib`, `seaborn` để trực quan hóa dữ liệu và feature importance.  
  - `Flask` để xây dựng giao diện web cho phép người dùng nhập dữ liệu và dự đoán trực tiếp.  
- **Công cụ**: Jupyter Notebook, Visual Studio Code/Anaconda.  

---

## 🖼️ Một số giao diện cơ bản

### 1. Giao diện nhập liệu dự đoán (web Flask)
Người dùng nhập các thông tin:
- **Tỷ lệ sử dụng tín dụng quay vòng**  
- **Số lần trễ hạn trên 90 ngày**  
- **Số người phụ thuộc**  
- **Tuổi**  
- **Thu nhập hàng tháng**  

<img width="632" height="757" alt="image" src="https://github.com/user-attachments/assets/29309311-1ffe-4f58-88cc-01572df06012" />

---

### 2. Kết quả dự đoán
Hiển thị kết quả ngay sau khi submit form:  

<img width="618" height="168" alt="image" src="https://github.com/user-attachments/assets/716c6344-6108-4689-bcaa-ff007d10b97e" />






