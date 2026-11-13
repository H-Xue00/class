#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int main() {
    // 1. 读取图像（兼容从 build/ 或项目根目录运行）
    string imgPath;
    for (const string& p : {string("../huixingzheng.jpg"), string("huixingzheng.jpg")}) {
        if (fs::exists(p)) {
            imgPath = p;
            break;
        }
    }
    if (imgPath.empty()) {
        cout << "错误：找不到 huixingzheng.jpg，请检查路径。" << endl;
        return -1;
    }

    Mat image = imread(imgPath, IMREAD_COLOR);
    if (image.empty()) {
        cout << "错误：无法读取图像: " << imgPath << endl;
        return -1;
    }

    // 复制原图用于最后画图展示
    Mat output_image = image.clone();

    // 2. 转换为灰度图
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    // 3. 高斯模糊去噪
    Mat blurred;
    GaussianBlur(gray, blurred, Size(5, 5), 0);

    // 4. 二值化 (Otsu自动阈值 + 反转)
    Mat thresh;
    threshold(blurred, thresh, 0, 255, THRESH_BINARY_INV | THRESH_OTSU);

    // 5. 形态学操作：闭运算 (填补回形针内部镂空)
    // 核的大小 (15, 15) 可根据实际分辨率调整
    Mat kernel = getStructuringElement(MORPH_RECT, Size(15, 15));
    Mat closed;
    morphologyEx(thresh, closed, MORPH_CLOSE, kernel);

    // 6. 寻找轮廓 (只寻找最外层轮廓)
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(closed, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 7. 过滤轮廓并计数
    int count = 0;
    double min_area = 500.0;   // 最小面积阈值，用于过滤噪点
    int border_margin = 10;    // 贴边容差：边框附近的干扰物（图像边缘暗纹等）不计数

    const int imgW = image.cols;
    const int imgH = image.rows;

    for (size_t i = 0; i < contours.size(); i++) {
        double area = contourArea(contours[i]);

        // 面积太小 -> 噪点
        if (area <= min_area) continue;

        // 贴边过滤：最左侧那条 7x796 的竖纹是图像边缘伪影，需排除
        Rect bb = boundingRect(contours[i]);
        bool touchesBorder = bb.x <= border_margin ||
                             bb.y <= border_margin ||
                             bb.x + bb.width  >= imgW - border_margin ||
                             bb.y + bb.height >= imgH - border_margin;
        if (touchesBorder) {
            cout << "  [干扰已排除] 贴边区域 bbox=(" << bb.x << "," << bb.y
                 << " " << bb.width << "x" << bb.height << ") 面积=" << (int)area << endl;
            continue;
        }

        count++;

        // 绘制轮廓 (绿色，线宽2)
        drawContours(output_image, contours, (int)i, Scalar(0, 255, 0), 2);

        // 计算轮廓中心点，用于标注序号
        Moments M = moments(contours[i]);
        if (M.m00 != 0) {
            int cX = int(M.m10 / M.m00);
            int cY = int(M.m01 / M.m00);

            // 在中心点写上序号 (红色)
            putText(output_image, to_string(count), Point(cX - 10, cY + 5),
                    FONT_HERSHEY_SIMPLEX, 0.6, Scalar(0, 0, 255), 2);
        }
    }

    // 8. 保存结果图片（替代 imshow，避免 WSLg 下 OpenCV 窗口黑屏）
    //    输出目录跟随图片所在目录：从 build/ 运行 -> ../result，从根目录运行 -> result
    const fs::path srcDir = fs::path(imgPath).parent_path();
    const fs::path outDir = srcDir.empty() ? fs::path("result") : srcDir / "result";
    fs::create_directories(outDir);

    const string p1 = (outDir / "1_original.png").string();
    const string p2 = (outDir / "2_threshold.png").string();
    const string p3 = (outDir / "3_closed.png").string();
    const string p4 = (outDir / "4_result.png").string();

    imwrite(p1, image);
    imwrite(p2, thresh);
    imwrite(p3, closed);
    imwrite(p4, output_image);

    cout << "检测到的回形针数量: " << count << endl;
    cout << "结果已保存到目录: " << fs::absolute(outDir).string() << endl;
    cout << "  " << p1 << endl;
    cout << "  " << p2 << endl;
    cout << "  " << p3 << endl;
    cout << "  " << p4 << endl;

    return 0;
}