#include <math.h>
#include <algorithm>

#define PI 3.1415926f 

struct Point {
  float x, y;
};

void Quad2Obb(const float* quad, float* obb) {
  Point p1 = { quad[0], quad[1] };
  Point p2 = { quad[2], quad[3] };
  Point p3 = { quad[4], quad[5] };
  // Point p4 = { quad[6], quad[7] };
  float edge_1 = sqrt(pow((p1.x - p2.x), 2) + pow((p1.y - p2.y), 2));
  float edge_2 = sqrt(pow((p2.x - p3.x), 2) + pow((p2.y - p3.y), 2));
  float theta = 0.f;
  float width, height;
  if (edge_1 > edge_2) {
    width = edge_1;
    height = edge_2;
    if (p1.x != p2.x)
      theta = -atan((p1.y - p2.y) / (p1.x - p2.x)) / PI * 180.f;
    else
      theta = 90.f;
  } else {
    width = edge_2;
    height = edge_1;
    if (p1.x != p3.x)
      theta = -atan((p2.y - p3.y) / (p2.x - p3.x)) / PI * 180.f;
    else
      theta = 90.f;
  }
  theta = theta < -45.f ? (theta + 180) : theta;
  float ctr_x = (p1.x + p3.x) / 2.f;
  float ctr_y = (p1.y + p3.y) / 2.f;
  obb[0] = ctr_x - width / 2.f;
  obb[1] = ctr_y - height / 2.f;
  obb[2] = ctr_x + width / 2.f;
  obb[3] = ctr_y + height / 2.f;
  obb[4] = theta;
}
