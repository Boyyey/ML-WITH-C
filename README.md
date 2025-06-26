# Linear Regression Temperature Predictor (C + SDL2) 🌡️📈

the screenshot:

![alt text](<Screenshot 2025-06-26 202238.png>)

## Overview
This project demonstrates **linear regression from scratch in C**, using gradient descent to predict temperature based on time of day. It features a modern, interactive GUI built with **SDL2** to visualize the data points and the regression line, making it easy to understand how the model fits the data.

- **Language:** C 💻
- **Graphics:** SDL2 🎨
- **Data Input:** CSV file (time, temperature) 📄
- **Features:**
  - Reads data from CSV 📥
  - Trains a linear regression model using gradient descent 🧠
  - Predicts and prints results 🔢
  - Plots data points and regression line with a clear, labeled GUI 🖼️

---

## 🚀 How to Build and Run (Windows, MSYS2/MinGW64, VS Code)

### 1️⃣ Install MSYS2 and SDL2
- Download and install [MSYS2](https://www.msys2.org/).
- Open the **MSYS2 MinGW64** terminal and run:
  ```sh
  pacman -Syu   # (update system, follow prompts)
  pacman -S mingw-w64-x86_64-gcc mingw-w64-x86_64-SDL2
  ```

### 2️⃣ Clone or Download This Project
- Place the files in a folder, e.g. `C:/Users/MEHR/OneDrive/Documents/Desktop/MLinC`

### 3️⃣ Open in VS Code
- Open the folder in VS Code.
- Make sure you have the C/C++ extension installed.

### 4️⃣ Configure Intellisense (Optional, for no red squiggles)
- `.vscode/c_cpp_properties.json` should include:
  ```json
  "includePath": [
    "${workspaceFolder}/**",
    "C:/msys64/mingw64/include/SDL2"
  ]
  ```

### 5️⃣ Build and Run
- Open a **MSYS2 MinGW64** terminal in the project folder and run:
  ```sh
  gcc linear_regression.c -o linear_regression -I/mingw64/include/SDL2 -L/mingw64/lib -lSDL2
  ./linear_regression.exe
  ```

---

## Data Format
- The program expects a CSV file named `data.csv` in the same folder, with two columns: `time,temperature` (no header).
  Example:
  ```
  0,12.5
  3,14.2
  6,16.8
  ...
  ```

## Features & Controls
- **Interactive GUI:**
  - Blue squares: Data points
  - Red line: Regression line
  - Labels, legend, and equation are clearly positioned
- **Controls:**
  - `Esc` or close window: Exit
  - `N`: Load new CSV data and retrain
  - `R`: Retrain on current data

---

## 📄 License
MIT License
