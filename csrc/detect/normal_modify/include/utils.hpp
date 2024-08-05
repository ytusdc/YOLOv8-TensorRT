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
