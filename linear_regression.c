// linear_regression.c
// Linear Regression from Scratch in C
// Predicts temperature based on time of day using gradient descent
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#define SDL_MAIN_HANDLED
#include <SDL2/SDL.h>

#define MAX_DATA 1000
#define LEARNING_RATE 0.0001
#define EPOCHS 10000

// Function to read CSV file
typedef struct {
    double x[MAX_DATA];
    double y[MAX_DATA];
    int size;
} Dataset;

Dataset read_csv(const char* filename) {
    Dataset data;
    data.size = 0;
    FILE* file = fopen(filename, "r");
    if (!file) {
        printf("Error opening file %s\n", filename);
        exit(1);
    }
    while (fscanf(file, "%lf,%lf", &data.x[data.size], &data.y[data.size]) == 2) {
        data.size++;
        if (data.size >= MAX_DATA) break;
    }
    fclose(file);
    return data;
}

// Mean Squared Error
double compute_mse(Dataset data, double m, double b) {
    double mse = 0.0;
    for (int i = 0; i < data.size; i++) {
        double y_pred = m * data.x[i] + b;
        mse += pow(y_pred - data.y[i], 2);
    }
    return mse / data.size;
}

// Gradient Descent
void train(Dataset data, double* m, double* b) {
    *m = 0.0;
    *b = 0.0;
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        double m_grad = 0.0;
        double b_grad = 0.0;
        for (int i = 0; i < data.size; i++) {
            double y_pred = (*m) * data.x[i] + (*b);
            m_grad += (y_pred - data.y[i]) * data.x[i];
            b_grad += (y_pred - data.y[i]);
        }
        m_grad = (2.0 / data.size) * m_grad;
        b_grad = (2.0 / data.size) * b_grad;
        *m -= LEARNING_RATE * m_grad;
        *b -= LEARNING_RATE * b_grad;
        if (epoch % 1000 == 0) {
            printf("Epoch %d: MSE = %f\n", epoch, compute_mse(data, *m, *b));
        }
    }
}

// Predict
double predict(double m, double b, double x) {
    return m * x + b;
}

// 5x7 bitmap font for A-Z, 0-9, space, and a few symbols
const unsigned char font5x7[44][7] = {
    // 0-9
    {0x1E,0x11,0x13,0x15,0x19,0x11,0x1E}, // 0
    {0x04,0x0C,0x04,0x04,0x04,0x04,0x0E}, // 1
    {0x1E,0x01,0x01,0x1E,0x10,0x10,0x1F}, // 2
    {0x1E,0x01,0x01,0x0E,0x01,0x01,0x1E}, // 3
    {0x02,0x06,0x0A,0x12,0x1F,0x02,0x02}, // 4
    {0x1F,0x10,0x1E,0x01,0x01,0x01,0x1E}, // 5
    {0x0E,0x10,0x1E,0x11,0x11,0x11,0x0E}, // 6
    {0x1F,0x01,0x02,0x04,0x08,0x08,0x08}, // 7
    {0x0E,0x11,0x11,0x0E,0x11,0x11,0x0E}, // 8
    {0x0E,0x11,0x11,0x0F,0x01,0x01,0x0E}, // 9
    // A-Z
    {0x0E,0x11,0x11,0x1F,0x11,0x11,0x11}, // A
    {0x1E,0x11,0x11,0x1E,0x11,0x11,0x1E}, // B
    {0x0E,0x11,0x10,0x10,0x10,0x11,0x0E}, // C
    {0x1E,0x11,0x11,0x11,0x11,0x11,0x1E}, // D
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x1F}, // E
    {0x1F,0x10,0x10,0x1E,0x10,0x10,0x10}, // F
    {0x0E,0x11,0x10,0x17,0x11,0x11,0x0E}, // G
    {0x11,0x11,0x11,0x1F,0x11,0x11,0x11}, // H
    {0x0E,0x04,0x04,0x04,0x04,0x04,0x0E}, // I
    {0x07,0x02,0x02,0x02,0x12,0x12,0x0C}, // J
    {0x11,0x12,0x14,0x18,0x14,0x12,0x11}, // K
    {0x10,0x10,0x10,0x10,0x10,0x10,0x1F}, // L
    {0x11,0x1B,0x15,0x15,0x11,0x11,0x11}, // M
    {0x11,0x19,0x15,0x13,0x11,0x11,0x11}, // N
    {0x0E,0x11,0x11,0x11,0x11,0x11,0x0E}, // O
    {0x1E,0x11,0x11,0x1E,0x10,0x10,0x10}, // P
    {0x0E,0x11,0x11,0x11,0x15,0x12,0x0D}, // Q
    {0x1E,0x11,0x11,0x1E,0x14,0x12,0x11}, // R
    {0x0F,0x10,0x10,0x0E,0x01,0x01,0x1E}, // S
    {0x1F,0x04,0x04,0x04,0x04,0x04,0x04}, // T
    {0x11,0x11,0x11,0x11,0x11,0x11,0x0E}, // U
    {0x11,0x11,0x11,0x11,0x11,0x0A,0x04}, // V
    {0x11,0x11,0x11,0x15,0x15,0x1B,0x11}, // W
    {0x11,0x11,0x0A,0x04,0x0A,0x11,0x11}, // X
    {0x11,0x11,0x0A,0x04,0x04,0x04,0x04}, // Y
    {0x1F,0x01,0x02,0x04,0x08,0x10,0x1F}, // Z
    // space, dash, dot, colon, comma, plus, equal
    {0x00,0x00,0x00,0x00,0x00,0x00,0x00}, // space
    {0x00,0x00,0x00,0x1F,0x00,0x00,0x00}, // dash
    {0x00,0x00,0x00,0x00,0x00,0x0C,0x0C}, // dot
    {0x00,0x04,0x00,0x00,0x04,0x00,0x00}, // colon
    {0x00,0x00,0x00,0x00,0x0C,0x0C,0x00}, // comma
    {0x00,0x04,0x0A,0x11,0x0A,0x04,0x00}, // plus
    {0x00,0x00,0x1F,0x00,0x1F,0x00,0x00}, // equal
};

// Map ASCII to font index
int font_index(char c) {
    if (c >= '0' && c <= '9') return c - '0';
    if (c >= 'A' && c <= 'Z') return c - 'A' + 10;
    if (c >= 'a' && c <= 'z') return c - 'a' + 10;
    if (c == ' ') return 36;
    if (c == '-') return 37;
    if (c == '.') return 38;
    if (c == ':') return 39;
    if (c == ',') return 40;
    if (c == '+') return 41;
    if (c == '=') return 42;
    return 36; // space for unsupported
}

void draw_text(SDL_Renderer* renderer, int x, int y, const char* text, SDL_Color color) {
    int i = 0;
    while (text[i]) {
        int idx = font_index(text[i]);
        for (int row = 0; row < 7; row++) {
            for (int col = 0; col < 5; col++) {
                if (font5x7[idx][row] & (1 << (4-col))) {
                    SDL_SetRenderDrawColor(renderer, color.r, color.g, color.b, color.a);
                    SDL_Rect dot = {x + i*6 + col, y + row, 1, 1};
                    SDL_RenderFillRect(renderer, &dot);
                }
            }
        }
        i++;
    }
}

void plot_regression_sdl(Dataset data, double m, double b) {
    const int win_w = 900, win_h = 650;
    SDL_Init(SDL_INIT_VIDEO);
    SDL_Window* window = SDL_CreateWindow("Temperature vs Time of Day - Linear Regression", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, win_w, win_h, 0);
    SDL_Renderer* renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    SDL_SetRenderDrawColor(renderer, 245, 255, 255, 255);
    SDL_RenderClear(renderer);
    SDL_Color black = {0,0,0,255};
    // Draw axes
    SDL_SetRenderDrawColor(renderer, 0, 0, 0, 255);
    SDL_RenderDrawLine(renderer, 70, win_h-70, win_w-70, win_h-70); // X axis
    SDL_RenderDrawLine(renderer, 70, 70, 70, win_h-70); // Y axis
    // Find min/max for scaling
    double min_x = data.x[0], max_x = data.x[0], min_y = data.y[0], max_y = data.y[0];
    for (int i = 1; i < data.size; i++) {
        if (data.x[i] < min_x) min_x = data.x[i];
        if (data.x[i] > max_x) max_x = data.x[i];
        if (data.y[i] < min_y) min_y = data.y[i];
        if (data.y[i] > max_y) max_y = data.y[i];
    }
    // Draw grid
    SDL_SetRenderDrawColor(renderer, 220, 220, 220, 255);
    for (int i = 1; i <= 10; i++) {
        int gx = 70 + i * (win_w-140) / 10;
        SDL_RenderDrawLine(renderer, gx, 70, gx, win_h-70);
        int gy = win_h-70 - i * (win_h-140) / 10;
        SDL_RenderDrawLine(renderer, 70, gy, win_w-70, gy);
    }
    // Axis labels and title (blocky text)
    // Center the title
    int title_w = 44 * 6; // 44 chars * 6px per char
    draw_text(renderer, (win_w-title_w)/2, 20, "Temperature Prediction using Linear Regression", black);
    // Center the x-axis label
    int xlabel_w = 12 * 6; // "Time of Day" is 12 chars
    draw_text(renderer, (win_w-xlabel_w)/2, win_h-40, "Time of Day", black);
    // Vertically center and rotate the y-axis label
    // Draw "Temperature" vertically
    int ylabel_len = 11; // "Temperature"
    for (int i = 0; i < ylabel_len; i++) {
        char c[2] = {"Temperature"[i], 0};
        draw_text(renderer, 20, win_h/2 - 35 + i*12, c, black);
    }
    // Plot data points
    SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255);
    for (int i = 0; i < data.size; i++) {
        int px = 70 + (int)((data.x[i] - min_x) / (max_x - min_x) * (win_w-140));
        int py = win_h-70 - (int)((data.y[i] - min_y) / (max_y - min_y) * (win_h-140));
        SDL_Rect pt = {px-4, py-4, 8, 8};
        SDL_RenderFillRect(renderer, &pt);
    }
    // Plot regression line (clip to axes)
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
    double t1 = min_x, t2 = max_x;
    double y1 = m * t1 + b;
    double y2 = m * t2 + b;
    // Clip y1/y2 to axis range
    if (y1 < min_y) { t1 = (min_y - b) / m; y1 = min_y; }
    if (y1 > max_y) { t1 = (max_y - b) / m; y1 = max_y; }
    if (y2 < min_y) { t2 = (min_y - b) / m; y2 = min_y; }
    if (y2 > max_y) { t2 = (max_y - b) / m; y2 = max_y; }
    int px1 = 70 + (int)((t1 - min_x) / (max_x - min_x) * (win_w-140));
    int px2 = 70 + (int)((t2 - min_x) / (max_x - min_x) * (win_w-140));
    int py1 = win_h-70 - (int)((y1 - min_y) / (max_y - min_y) * (win_h-140));
    int py2 = win_h-70 - (int)((y2 - min_y) / (max_y - min_y) * (win_h-140));
    SDL_RenderDrawLine(renderer, px1, py1, px2, py2);
    // Draw legend (move to top left)
    int legend_x = 90, legend_y = 80;
    SDL_SetRenderDrawColor(renderer, 0, 0, 255, 255);
    SDL_Rect legend_data = {legend_x, legend_y, 20, 20};
    SDL_RenderFillRect(renderer, &legend_data);
    draw_text(renderer, legend_x+30, legend_y, "Data Points", black);
    SDL_SetRenderDrawColor(renderer, 255, 0, 0, 255);
    SDL_RenderDrawLine(renderer, legend_x, legend_y+40, legend_x+20, legend_y+40);
    draw_text(renderer, legend_x+30, legend_y+30, "Regression Line", black);
    // Draw equation (move below legend)
    char eqn[128];
    snprintf(eqn, sizeof(eqn), "y = %.2fx + %.2f", m, b);
    draw_text(renderer, legend_x, legend_y+60, eqn, black);
    // Draw instructions (centered at bottom)
    const char* instr = "[Esc] or [Close] to exit | [N] New Data | [R] Retrain";
    int instr_w = 56 * 6; // 56 chars * 6px
    draw_text(renderer, (win_w-instr_w)/2, win_h-30, instr, black);
    SDL_RenderPresent(renderer);
    // GUI event loop
    int quit = 0;
    while (!quit) {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_QUIT) quit = 1;
            if (e.type == SDL_KEYDOWN) {
                if (e.key.keysym.sym == SDLK_ESCAPE) quit = 1;
                if (e.key.keysym.sym == SDLK_n) {
                    // Prompt for new CSV file
                    char newfile[256];
                    printf("Enter new CSV filename: ");
                    scanf("%255s", newfile);
                    Dataset newdata = read_csv(newfile);
                    double nm, nb;
                    train(newdata, &nm, &nb);
                    plot_regression_sdl(newdata, nm, nb);
                    quit = 1;
                }
                if (e.key.keysym.sym == SDLK_r) {
                    // Retrain on same data
                    double nm, nb;
                    train(data, &nm, &nb);
                    plot_regression_sdl(data, nm, nb);
                    quit = 1;
                }
            }
        }
        SDL_Delay(10);
    }
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

int main(int argc, char* argv[]) {
    const char* filename = "data.csv";
    Dataset data = read_csv(filename);
    double m, b;
    train(data, &m, &b);
    printf("\nTrained model: y = %.4fx + %.4f\n", m, b);
    printf("\nPredictions:\n");
    for (int i = 0; i < data.size; i++) {
        double y_pred = predict(m, b, data.x[i]);
        printf("Time: %.2f, Actual Temp: %.2f, Predicted Temp: %.2f\n", data.x[i], data.y[i], y_pred);
    }
    plot_regression_sdl(data, m, b);
    return 0;
}
