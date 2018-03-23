#include <math.h>
#include <algorithm>

using std::max;
using std::min;
using std::abs;

#define PI 3.1415926f
#define NUM_PTS 4

struct Point {
  float x, y;
};

void SortPoints(float* pts) {
  // sort points by clockwise
  float tmp[NUM_PTS * 2];
  float x = 0.f, y = 0.f;
  for (int i = 0; i < NUM_PTS; ++i) {
    x += pts[i * 2];
    y += pts[i * 2 + 1];
    tmp[i * 2] = pts[i * 2];
    tmp[i * 2 + 1] = pts[i * 2 + 1];
  }
  Point center = { x / NUM_PTS, y / NUM_PTS };
  for (int i = 0; i < NUM_PTS; ++i) {
    tmp[i * 2] = tmp[i * 2] - center.x;
    tmp[i * 2 + 1] = tmp[i * 2 + 1] - center.y;
  }
  float theta[NUM_PTS];
  float val = 0.f;
  for (int i = 0; i < NUM_PTS; ++i) {
    x = tmp[i * 2];
    y = tmp[i * 2 + 1];
    // occur NaN on Ubuntu
    // make sure value between -1 and 1
    val = (x * 1.f + y * 0.f) / (sqrt(x * x + y * y) * 1.f);
    val = min(max(val, -1.f), 1.f);
    theta[i] = acos(val) / PI * 180.f;
    theta[i] = y > 0 ? 360.f - theta[i] : theta[i];
  }
  float sorted_theta[NUM_PTS];
  for (int i = 0; i < NUM_PTS; ++i) {
    sorted_theta[i] = theta[i];
  }
  // bubble sort and clockwise
  for (int i = 0; i < (NUM_PTS - 1); ++i) {
    for (int j = 0; j < (NUM_PTS - i - 1); ++j) {
      if (sorted_theta[j] < sorted_theta[j + 1]) {
        float temp = sorted_theta[j];
        sorted_theta[j] = sorted_theta[j + 1];
        sorted_theta[j + 1] = temp;
      }
    }
  }
  for (int i = 0; i < NUM_PTS; ++i) {
    tmp[i * 2] = tmp[i * 2] + center.x;
    tmp[i * 2 + 1] = tmp[i * 2 + 1] + center.y;
  }
  for (int i = 0; i < NUM_PTS; ++i) {
    float t = sorted_theta[i];
    int idx = -1;
    for (int j = 0; j < NUM_PTS; ++j) {
      if (theta[j] == t) {
        idx = j;
        break;
      }
    }
    pts[i * 2] = tmp[idx * 2];
    pts[i * 2 + 1] = tmp[idx * 2 + 1];
  }
  // reference from [2017 CVPR] DMPN
  // 'From the line with bigger slope, choose the point with smaller x as the new first point'
  float slope_1 = (pts[5] - pts[1]) / (pts[4] - pts[0]);
  float slope_2 = (pts[7] - pts[3]) / (pts[6] - pts[2]);
  int first_idx = -1;
  if (slope_1 > slope_2) {
    if (pts[0] < pts[4])
      first_idx = 0;
    else if (pts[0] == pts[4])
      first_idx = pts[1] < pts[5] ? 0 : 2;
    else
      first_idx = 2;
  } else {
    if (pts[2] < pts[6])
      first_idx = 1;
    else if (pts[2] == pts[6])
      first_idx = pts[3] < pts[7] ? 1 : 3;
    else
      first_idx = 3;
  }
  for (int i = 0; i < NUM_PTS; ++i) {
    tmp[i * 2] = pts[i * 2];
    tmp[i * 2 + 1] = pts[i * 2 + 1];
  }
  for (int i = 0; i < NUM_PTS; ++i) {
    int idx = (first_idx + i) % NUM_PTS;
    pts[i * 2] = tmp[idx * 2];
    pts[i * 2 + 1] = tmp[idx * 2 + 1];
  }
}

void SortPoints(const float* in, float* out) {
  for (int i = 0; i < NUM_PTS; ++i) {
    out[i] = in[i];
  }
  SortPoints(out);
}
