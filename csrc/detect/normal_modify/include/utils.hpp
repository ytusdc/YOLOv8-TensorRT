#define min(a,b) ((a)<(b)?(a):(b))

struct AffineMatrix {

	float i2d[6],d2i[6];

	void compute(const int from_width, const int from_height,const int to_width, const int to_height) {
		float scale_x = to_width / (float)from_width;
		float scale_y = to_height / (float)from_height;

		float scale = min(scale_x, scale_y);
		float ox = (-scale * from_width + to_width + scale - 1) * 0.5;
		float oy = (-scale * from_height + to_height + scale - 1) * 0.5;
		float k = 1 / scale;

		i2d[0] = scale;  i2d[1] = 0; i2d[2] = ox;
		i2d[3] = 0;  i2d[4] = scale;  i2d[5] = oy;

		d2i[0] = k;  d2i[1] = 0; d2i[2] = -k*ox;
		d2i[3] = 0;  d2i[4] = k;  d2i[5] = -k*oy;

	}

};

// struct Size{
//     int width = 0, height = 0;

//     Size() = default;
//     Size(int w, int h)
//     :width(w), height(h){}
// };

// struct AffineMatrix{
//     float i2d[6];       // image to dst(network), 2x3 matrix ==> M
//     float d2i[6];       // dst to image, 2x3 matrix ==> M^-1

//     // 求解M的逆矩阵，由于这个3x3矩阵的第三行是确定的0, 0, 1，因此可以简写如下
//     void invertAffineTransform(float imat[6], float omat[6]){
//         float i00 = imat[0]; float i01 = imat[1]; float i02 = imat[2];
//         float i10 = imat[3]; float i11 = imat[4]; float i12 = imat[5];

//         // 计算行列式
//         float D = i00 * i11 - i01 * i10;
//         D = D != 0 ? 1.0 / D : 0;
        
//         // 计算剩余的伴随矩阵除以行列式
//         float A11 = i11 * D;
//         float A22 = i00 * D;
//         float A12 = -i01 * D;
//         float A21 = -i10 * D;
//         float b1 = -A11 * i02 - A12 * i12;
//         float b2 = -A21 * i02 - A22 * i12;
//         omat[0] = A11; omat[1] = A12; omat[2] = b1;
//         omat[3] = A21; omat[4] = A22; omat[5] = b2;
//     }

//     // 求解M矩阵
//     void compute(const Size& from, const Size& to){
//         float scale_x = to.width / (float)from.width;
//         float scale_y = to.height / (float)from.height;

//         float scale = min(scale_x, scale_y);
//         /*
//         M = [
//         scale,    0,     -scale * from.width  * 0.5 + to.width  * 0.5
//         0,     scale,    -scale * from.height * 0.5 + to.height * 0.5
//         0,        0,                     1
//         ]
//         */
//         /*
//             - 0.5 的主要原因是使得中心更加对齐，下采样不明显，但是上采样时就比较明显
//             参考：https://www.iteye.com/blog/handspeaker-1545126
//         */
//         i2d[0] = scale; i2d[1] = 0; i2d[2] = 
//             -scale * from.width  * 0.5 + to.width  * 0.5 + scale * 0.5 - 0.5;
        
//         i2d[3] = 0; i2d[4] = scale; i2d[5] =
//             -scale * from.height * 0.5 + to.height * 0.5 + scale * 0.5 - 0.5;
        
//         invertAffineTransform(i2d, d2i);
//     }
// };
