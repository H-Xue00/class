// 圆孔检测：CLAHE 增强 + 霍夫圆变换
// 输入：yuankong.jpg   输出：result/hole_*.png
#include <opencv2/opencv.hpp>
#include <iostream>
#include <vector>
#include <filesystem>
#include <algorithm>

using namespace cv;
using namespace std;
namespace fs = std::filesystem;

int main() {
    // 1. 读图（兼容从 build/ 或项目根目录运行）
    string imgPath;
    for (const string& p : {string("../yuankong.jpg"), string("yuankong.jpg")}) {
        if (fs::exists(p)) {
            imgPath = p;
            break;
        }
    }
    if (imgPath.empty()) {
        cout << "错误：找不到 yuankong.jpg，请检查路径。" << endl;
        return -1;
    }

    Mat image = imread(imgPath, IMREAD_COLOR);
    if (image.empty()) {
        cout << "错误：无法读取图像: " << imgPath << endl;
        return -1;
    }

    Mat output_image = image.clone();

    // 2. 灰度 + CLAHE 局部对比度增强
    //    原图孔与背景灰度差很小（约 20 级），直接阈值不可靠，用 CLAHE 拉开差距
    Mat gray;
    cvtColor(image, gray, COLOR_BGR2GRAY);

    Ptr<CLAHE> clahe = createCLAHE(3.0, Size(8, 8));
    Mat enhanced;
    clahe->apply(gray, enhanced);

    // 3. 高斯模糊抑制噪声（霍夫圆对噪声敏感）
    Mat blurred;
    GaussianBlur(enhanced, blurred, Size(9, 9), 2.0);

    // 4. 霍夫圆变换
    //    dp=1           与原图同分辨率
    //    minDist=60     两圆心最小间距，避免同一孔被重复检出
    //    param1=120     Canny 高阈值
    //    param2=28      累加器阈值，越大越严格
    //    minRadius/maxRadius  限定小圆孔，排除右侧大孔（半径约 150）
    vector<Vec3f> circles;
    HoughCircles(blurred, circles, HOUGH_GRADIENT,
                 /*dp=*/1, /*minDist=*/60,
                 /*param1=*/120, /*param2=*/28,
                 /*minRadius=*/15, /*maxRadius=*/60);

    cout << "霍夫检测到候选圆: " << circles.size() << " 个" << endl;

    // 5. 按面积排序后编号（面积从大到小），保证输出稳定
    sort(circles.begin(), circles.end(), [](const Vec3f& a, const Vec3f& b) {
        return a[2] * a[2] > b[2] * b[2];
    });

    int count = 0;
    for (const Vec3f& c : circles) {
        Point center(cvRound(c[0]), cvRound(c[1]));
        int radius = cvRound(c[2]);

        count++;

        // 绿色圆圈
        circle(output_image, center, radius, Scalar(0, 255, 0), 2);
        // 黄色圆心
        circle(output_image, center, 2, Scalar(0, 255, 255), 3);

        // 红色编号（圆圈上方）
        putText(output_image, to_string(count), Point(center.x - 10, center.y - radius - 10),
                FONT_HERSHEY_SIMPLEX, 0.8, Scalar(0, 0, 255), 2);
        // 半径标注（圆圈下方）
        putText(output_image, "r=" + to_string(radius), Point(center.x - 22, center.y + radius + 22),
                FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 0, 255), 1);

        cout << "  圆孔 #" << count << " 圆心=(" << center.x << "," << center.y
             << ") 半径=" << radius << endl;
    }

    cout << "检测到的圆孔数量: " << count << endl;

    // 6. 保存结果
    const fs::path srcDir = fs::path(imgPath).parent_path();
    const fs::path outDir = srcDir.empty() ? fs::path("result") : srcDir / "result";
    fs::create_directories(outDir);

    const string p1 = (outDir / "hole_1_enhanced.png").string();
    const string p2 = (outDir / "hole_2_result.png").string();

    imwrite(p1, enhanced);
    imwrite(p2, output_image);

    cout << "结果已保存到目录: " << fs::absolute(outDir).string() << endl;
    cout << "  " << p1 << endl;
    cout << "  " << p2 << endl;

    return 0;
}
