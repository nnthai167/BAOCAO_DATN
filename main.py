import sys
import os
import numpy as np
import pandas as pd
import joblib
from scipy.fft import fft
from PyQt5 import QtWidgets
from EQ5 import Ui_MainWindow

from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.backends.backend_qt5agg import NavigationToolbar2QT as NavigationToolbar
from matplotlib.figure import Figure


class MainApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        # Vẽ biểu đồ
        self.canvas = FigureCanvas(Figure(figsize=(30, 15)))
        self.canvas.setParent(self.ui.groupBox)
        self.canvas.setGeometry(40, 30, 590, 300)

        self.toolbar = NavigationToolbar(self.canvas, self)
        self.toolbar.setParent(self.ui.groupBox)
        self.toolbar.setGeometry(10, 0, 500, 25)

        # ComboBox & Button
        self.ui.channel.addItems(["x", "y", "z"])
        self.ui.type_chart.addItems(["Time_acceleration", "FFT_Amplitude"])

        self.select_file_btn = QtWidgets.QPushButton("Chọn file CSV", self.ui.centralwidget)
        self.select_file_btn.setGeometry(520, 450, 165, 23)
        self.select_file_btn.setStyleSheet("background-color: white; font-weight: bold;")
        self.select_file_btn.clicked.connect(self.choose_csv_file)

        self.ui.run.clicked.connect(self.plot_chart)
        self.current_csv_file = None

    def choose_csv_file(self):
        file_path, _ = QtWidgets.QFileDialog.getOpenFileName(self, "Chọn file CSV", "", "CSV Files (*.csv)")
        if file_path:
            self.current_csv_file = file_path
            self.ui.label_8.setText(f"Đã chọn: {os.path.basename(file_path)}")
        else:
            self.ui.label_8.setText("Chưa chọn file CSV.")

    def zero_crossing_rate(self, signal):
        return np.sum(np.diff(np.sign(signal)) != 0)

    def interquartile_range(self, signal):
        return np.percentile(signal, 75) - np.percentile(signal, 25)

    def extract_features_from_window(self, df):
        features = {}
        for axis in ['x', 'y', 'z']:
            data = df[axis].values
            features[f'IQR_{axis}'] = self.interquartile_range(data)
            features[f'ZC_{axis}'] = self.zero_crossing_rate(data)
            fft_values = np.abs(fft(data))[:len(data) // 2]
            features[f'Dominant_freq_{axis}'] = np.argmax(fft_values)
            features[f'Energy_{axis}'] = np.sum(fft_values**2)
        return features

    def predict_eq_from_csv(self):
        try:
            df = pd.read_csv(self.current_csv_file)
            if not all(col in df.columns for col in ['x', 'y', 'z']):
                raise ValueError("Thiếu cột x, y, z!")

            # Cấu hình chia cửa sổ
            sampling_rate = 100
            window_duration = 10
            samples_per_segment = sampling_rate * window_duration
            overlap = 0.5
            step_size = int(samples_per_segment * (1 - overlap))

            segments = []
            for start in range(0, len(df) - samples_per_segment + 1, step_size):
                segment = df.iloc[start:start + samples_per_segment]
                features = self.extract_features_from_window(segment)
                segments.append(features)

            if not segments:
                raise ValueError("Dữ liệu quá ngắn!")

            feature_df = pd.DataFrame(segments)
            features_to_use = ['Dominant_freq_y', 'Energy_z', 'ZC_z', 'ZC_y', 'IQR_z']

            X = feature_df[features_to_use]

            # Load model phân loại EQ
            model_clf = joblib.load("H:/FInal_exam/RD_model.pkl")
            scaler_clf = joblib.load("H:/FInal_exam/RD_scaler.pkl")
            X_scaled = scaler_clf.transform(X)
            y_pred = model_clf.predict(X_scaled)
            y_proba = model_clf.predict_proba(X_scaled)[:, 1]

            eq_ratio = np.mean(y_pred)
            is_eq = eq_ratio >= 0.5
            label = "🌋 EQ" if is_eq else "🌬️ NonEQ"
            color = "red" if is_eq else "green"

            self.ui.result.setText(label)
            self.ui.result.setStyleSheet(f"background-color: white; color: {color};")

            # Nếu là EQ → Dự đoán độ lớn
            if is_eq:
                model_reg = joblib.load("H:\FInal_exam\MAG_model (1).pkl")
                scaler_reg = joblib.load("H:\FInal_exam\MAG_scaler (1).pkl")
                X_reg = scaler_reg.transform(X)
                mag_preds = model_reg.predict(X_reg)
                avg_mag = np.mean(mag_preds)
                self.ui.label_11.setText(f"{avg_mag:.2f}")
                if avg_mag >= 6:
                    self.ui.label_11.setStyleSheet("background-color: red; color: white;")
                elif avg_mag >= 4.5:
                    self.ui.label_11.setStyleSheet("background-color: orange; color: black;")
                else:
                    self.ui.label_11.setStyleSheet("background-color: yellow; color: black;")


            else:
                self.ui.label_11.setText("N/A")

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi", f"Lỗi khi dự đoán EQ hoặc độ lớn: {str(e)}")


        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi", f"Lỗi khi dự đoán EQ: {str(e)}")

    def plot_chart(self):
        if not self.current_csv_file or not os.path.exists(self.current_csv_file):
            QtWidgets.QMessageBox.warning(self, "Lỗi", "Vui lòng chọn file CSV hợp lệ trước.")
            return

        selected_axis = self.ui.channel.currentText()
        chart_type = self.ui.type_chart.currentText()

        try:
            df = pd.read_csv(self.current_csv_file)

            if selected_axis not in df.columns:
                QtWidgets.QMessageBox.warning(self, "Lỗi", f"Không tìm thấy cột '{selected_axis}' trong file CSV.")
                return

            if 'timestamp' not in df.columns:
                QtWidgets.QMessageBox.warning(self, "Lỗi", "File CSV cần có cột 'timestamp'.")
                return

            self.canvas.figure.clf()
            ax = self.canvas.figure.add_subplot(111)

            if chart_type == "Time_acceleration":
                ax.plot(df['timestamp'], df[selected_axis], label=f'{selected_axis.upper()} Axis', color='blue')
                ax.set_title(f"Biểu đồ gia tốc theo thời gian ({selected_axis.upper()})")
                ax.set_xlabel("Time (s)")
                ax.set_ylabel("Acceleration (g)")
                ax.legend()
            elif chart_type == "FFT_Amplitude":
                from scipy.fft import rfft, rfftfreq
                signal = df[selected_axis].values
                N = len(signal)
                T = (df['timestamp'].iloc[-1] - df['timestamp'].iloc[0]) / N
                yf = rfft(signal)
                xf = rfftfreq(N, T)

                ax.plot(xf, abs(yf), color='red')
                ax.set_title(f"Phổ FFT của trục {selected_axis.upper()}")
                ax.set_xlabel("Tần số (Hz)")
                ax.set_ylabel("Biên độ")

            self.canvas.draw()

            # Dự đoán sau khi vẽ
            self.predict_eq_from_csv()

        except Exception as e:
            QtWidgets.QMessageBox.critical(self, "Lỗi", f"Không thể đọc/vẽ biểu đồ:\n{str(e)}")


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainApp()
    window.show()
    sys.exit(app.exec_())
