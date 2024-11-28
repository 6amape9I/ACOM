import org.opencv.core.Core;
import org.opencv.core.CvType;
import org.opencv.core.Mat;
import org.opencv.highgui.HighGui;
import org.opencv.imgcodecs.Imgcodecs;

public class GaussianBlurExample {

    static {
        // Load the OpenCV native library
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public static void main(String[] args) {
        // Read the image in grayscale
        Mat img = Imgcodecs.imread("./img.png", Imgcodecs.IMREAD_GRAYSCALE);

        if (img.empty()) {
            System.out.println("Could not open or find the image!");
            return;
        }

        int coreSize = 15; // Example core size
        double standardDeviation = 10.0; // Example standard deviation

        // Apply Gaussian blur
        Mat blurredImage = gausBlur(img, coreSize, standardDeviation);

        // Display the original and blurred images
        HighGui.imshow("Original Image", img);
        HighGui.imshow("Blurred Image", blurredImage);
        HighGui.waitKey(0); // Wait for a key press
    }

    public static Mat gausBlur(Mat img, int coreSize, double standardDeviation) {
        Mat core = new Mat(coreSize, coreSize, CvType.CV_64F);

        // Fill the kernel with Gaussian values
        fillMatrix(core, coreSize, standardDeviation);

        // Normalize the kernel
        normMatrix(core);

        // Create an output image
        Mat outputImage = new Mat(img.size(), img.type());

        int xStart = coreSize / 2;
        int yStart = coreSize / 2;

        for (int i = xStart; i < img.rows() - xStart; i++) {
            for (int j = yStart; j < img.cols() - yStart; j++) {
                double val = 0;
                for (int k = -xStart; k <= xStart; k++) {
                    for (int l = -yStart; l <= yStart; l++) {
                        val += img.get(i + k, j + l)[0] * core.get(k + xStart, l + yStart)[0];
                    }
                }
                outputImage.put(i, j, val);
            }
        }

        return outputImage;
    }

    public static void fillMatrix(Mat core, int coreSize, double standardDeviation) {
        int a = (coreSize + 1) / 2;
        int b = a;

        for (int i = 0; i < coreSize; i++) {
            for (int j = 0; j < coreSize; j++) {
                double value = gauss(i, j, standardDeviation, a, b);
                core.put(i, j, value);
            }
        }
    }

    public static double gauss(int x, int y, double omega, int a, int b) {
        double omega2 = 2 * omega * omega;

        double m1 = 1 / (Math.PI * omega2);
        double m2 = Math.exp(-((x - a) * (x - a) + (y - b) * (y - b)) / omega2);

        return m1 * m2;
    }

    public static void normMatrix(Mat core) {
        double sum = 0;

        // Calculate the sum of all elements in the matrix
        for (int i = 0; i < core.rows(); i++) {
            for (int j = 0; j < core.cols(); j++) {
                sum += core.get(i, j)[0];
            }
        }

        // Normalize each element in the matrix
        for (int i = 0; i < core.rows(); i++) {
            for (int j = 0; j < core.cols(); j++) {
                double normalizedValue = core.get(i, j)[0] / sum;
                core.put(i, j, normalizedValue);
            }
        }
    }
}