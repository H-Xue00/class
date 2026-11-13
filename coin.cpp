// 硬币计数：阈值分割 + 轮廓过滤（面积 / 圆度）
// 输入：yingbi.jpg   输出：result/coin_*.png
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int main() {
    // 1. 读图（兼容从 build/ 或项目根目录运行）
    string imgPath;
    for (const string& p : {string("../yingbi.jpg"), string("yingbi.jpg")}) {
        if (fs::exists(p)) {
            imgPath = p;
            break;
        }
    }
    if (imgPath.empty()) {
        cout << "错误：找不到 yingbi.jpg，请检查路径。" << endl;
        return -1;
    }

    Mat image = imread(imgPath, IMREAD_COLOR);
    if (image.empty()) {
        cout << "错误：无法读取图像: " << imgPath << endl;
        return -1;
    }

    Mat output_image = image.clone();

    // 2. 灰度 + 高斯模糊
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);
    Mat blurred;
    GaussianBlur(gray, blurred, Size(7, 7), 1.5);

    // 3. 二值化：硬币是亮的前景，背景很暗
    //    Otsu 自动阈值，正片（亮=白），不需要反转
    Mat thresh;
    threshold(blurred, thresh, 0, 255, THRESH_BINARY | THRESH_OTSU);

    // 4. 形态学开运算：去掉噪点；小核闭运算：填补硬币内部细小空洞
    //    注意核不能太大，否则相邻硬币会被粘连成一个区域
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(5, 5));
    Mat cleaned;
    morphologyEx(thresh, cleaned, MORPH_OPEN, kernel);
    morphologyEx(cleaned, cleaned, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(7, 7)));

    // 4.1 加黑边框：避免贴着图像边缘的硬币与边界合并成一个大轮廓而漏检
    const int PAD = 30;
    Mat padded;
    copyMakeBorder(cleaned, padded, PAD, PAD, PAD, PAD, BORDER_CONSTANT, Scalar(0));

    // 5. 找外轮廓
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    findContours(padded, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    // 6. 面积 / 圆度过滤后计数
    const double min_area = 800.0;   // 过滤小噪点
    const double max_area = 60000.0; // 过滤过大区域
    const double min_circularity = 0.70; // 4*pi*A/P^2，圆形≈1

    int count = 0;
    for (size_t i = 0; i < contours.size(); i++) {
        // 用凸包判断形状：暗硬币内部有空洞时会变成环形，凸包可以还原成圆
        vector<Point> hull;
        convexHull(contours[i], hull);
        double area = contourArea(hull);
        if (area < min_area || area > max_area) continue;

        double peri = arcLength(hull, true);
        if (peri <= 0) continue;
        double circularity = 4.0 * CV_PI * area / (peri * peri);
        if (circularity < min_circularity) continue;

        count++;
        cout << "  硬币 #" << count << " 面积=" << (int)area
             << " 圆度=" << circularity << endl;

        // 坐标减去 PAD 才是原图位置
        vector<Point> shifted;
        shifted.reserve(hull.size());
        for (const Point& p : hull) shifted.emplace_back(p.x - PAD, p.y - PAD);

        // 绿色轮廓
        vector<vector<Point>> one{shifted};
        drawContours(output_image, one, 0, Scalar(0, 255, 0), 2);

        // 画最小外接圆，直观显示检测到的圆形
        Point2f center;
        float radius;
        minEnclosingCircle(shifted, center, radius);
        circle(output_image, center, (int)radius, Scalar(0, 255, 255), 1);

        // 序号（红色）
        Moments M = moments(shifted);
        if (M.m00 != 0) {
            int cX = int(M.m10 / M.m00);
            int cY = int(M.m01 / M.m00);
            putText(output_image, to_string(count), Point(cX - 10, cY + 8),
                    FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
        }
    }

    // 7. 保存结果
    const fs::path srcDir = fs::path(imgPath).parent_path();
    const fs::path outDir = srcDir.empty() ? fs::path("result") : srcDir / "result";
    fs::create_directories(outDir);

    const string p1 = (outDir / "coin_1_threshold.png").string();
    const string p2 = (outDir / "coin_2_cleaned.png").string();
    const string p3 = (outDir / "coin_3_result.png").string();

    imwrite(p1, thresh);
    imwrite(p2, cleaned);
    imwrite(p3, output_image);

    cout << "检测到的硬币数量: " << count << endl;
    cout << "结果已保存到目录: " << fs::absolute(outDir).string() << endl;
    cout << "  " << p1 << endl;
    cout << "  " << p2 << endl;
    cout << "  " << p3 << endl;

    return 0;
}
