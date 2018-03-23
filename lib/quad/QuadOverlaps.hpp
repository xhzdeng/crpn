#include <math.h>
#include <algorithm>

using std::max;
using std::min;
using std::abs;

// #define CUDA
#ifndef CUDA
#define PREFIX 
#else
#define PREFIX __device__
#endif

#define PI 3.1415926f  
#define EPS 1e-6
#define MAX_NUM 10
#define INVALID_VAL (-1.f)

struct Point {
  float x, y;
};

PREFIX inline int Sgn(const float x) {
  return x < -EPS ? -1 : (x > EPS);
}

PREFIX inline float Cross(const Point& p1, const Point& p2, const Point& p3, const Point& p4) {
  return (p2.x - p1.x) * (p4.y - p3.y) - (p2.y - p1.y) * (p4.x - p3.x);
}

PREFIX float Area(const float* pts, const int num) {
  float a = 0.f;
  for (int i = 0; i < num; ++i) {
    int j = (i + 1) % num;
    a += pts[i * 2] * pts[j * 2 + 1];
    a -= pts[i * 2 + 1] * pts[j * 2];
  }
  a = a / 2.f;
  return (a < 0.f ? -a : a);
}

PREFIX inline float Area(const Point& p1, const Point& p2, const Point& p3) {
  return abs(Cross(p1, p2, p1, p3)) / 2.f;
}

PREFIX inline float Median(float a, float b, float c) {
  if (a < b) {
    if (b < c)
      return b;
    else if (a < c)
      return c;
    else
      return a;
  } else if (a < c)
    return a;
  else if (b < c)
    return c;
  else
    return b;
}

PREFIX inline Point Median(const Point& p1, const Point& p2, const Point& p3) {
  Point p = { Median(p1.x, p2.x, p3.x), Median(p1.y, p2.y, p3.y) };
  return p;
}

PREFIX inline int Length(const float* p, const int max_size) {
  int rtn = max_size;
  for (int i = 0; i < max_size; ++i) {
    if (p[i * 2] == INVALID_VAL) {
      rtn = i;
      break;
    }
  }
  return rtn;
}

PREFIX inline bool Intersect(const Point& p1, const Point& p2, const Point& p3, const Point& p4) {
  return max(min(p1.x, p2.x), min(p3.x, p4.x)) <= min(max(p1.x, p2.x), max(p3.x, p4.x)) &&
         max(min(p1.y, p2.y), min(p3.y, p4.y)) <= min(max(p1.y, p2.y), max(p3.y, p4.y)) &&
         Sgn(Cross(p3, p2, p3, p4) * Cross(p3, p4, p3, p1)) >= 0.f && 
         Sgn(Cross(p1, p4, p1, p2) * Cross(p1, p2, p1, p3)) >= 0.f;
}

PREFIX void Intersect(const Point& p1, const Point& p2, const Point& p3, const Point& p4, float* out) {
  for (int i = 0; i < 4; ++i)
    out[i] = INVALID_VAL;
  if (Intersect(p1, p2, p3, p4)) {
    float a1 = Area(p1, p2, p3);
    float a2 = Area(p1, p2, p4);
    if (a1 == 0.f || a2 == 0.f) {
      if (a1 == 0.f) {
        Point md = Median(p1, p2, p3);
        out[0] = md.x;
        out[1] = md.y;
      }
      if (a2 == 0.f) {
        Point md = Median(p1, p2, p4);
        int idx = Length(out, 2);
        out[idx * 2] = md.x;
        out[idx * 2 + 1] = md.y;
      }
    } else {
      float k = a1 / a2;
      out[0] = (p3.x + k * p4.x) / (1 + k);
      out[1] = (p3.y + k * p4.y) / (1 + k);
    }
  }
}

PREFIX inline bool Contains(const Point& pt, const Point& A, const Point& B, const Point& C, const Point& D) {
  float a = (B.x - A.x) * (pt.y - A.y) - (B.y - A.y) * (pt.x - A.x);
  float b = (C.x - B.x) * (pt.y - B.y) - (C.y - B.y) * (pt.x - B.x);
  float c = (D.x - C.x) * (pt.y - C.y) - (D.y - C.y) * (pt.x - C.x);
  float d = (A.x - D.x) * (pt.y - D.y) - (A.y - D.y) * (pt.x - D.x);
  if ((a > 0.f && b > 0.f && c > 0.f && d > 0.f) || (a < 0.f && b < 0.f && c < 0.f && d < 0.f))
    return true;
  else
    return false;
}

PREFIX inline bool Contains(const Point& pt, const float* quad) {
  Point A = { quad[0], quad[1] };
  Point B = { quad[2], quad[3] };
  Point C = { quad[4], quad[5] };
  Point D = { quad[6], quad[7] };
  return Contains(pt, A, B, C, D);
}

PREFIX void SortPoints(float* pts, const int num) {
  if (num < 2)
    return;
  float tmp[MAX_NUM * 2];
  float x = 0.f, y = 0.f;
  for (int i = 0; i < num; ++i) {
    x += pts[i * 2];
    y += pts[i * 2 + 1];
    tmp[i * 2] = pts[i * 2];
    tmp[i * 2 + 1] = pts[i * 2 + 1];
  }
  Point center = { x / num, y / num };
  for (int i = 0; i < num; ++i) {
    tmp[i * 2] = tmp[i * 2] - center.x;
    tmp[i * 2 + 1] = tmp[i * 2 + 1] - center.y;
  }
  float theta[MAX_NUM];
  float val = 0.f;
  for (int i = 0; i < num; ++i) {
    x = tmp[i * 2];
    y = tmp[i * 2 + 1];
    // # occur NaN on Ubuntu
    // make sure value between -1 and 1
    val = (x * 1.f + y * 0.f) / (sqrt(x * x + y * y) * 1.f);
    val = min(max(val, -1.f), 1.f);
    theta[i] = acos(val) / PI * 180.f;
    theta[i] = y > 0 ? 360.f - theta[i] : theta[i];
  }
  float sorted_theta[MAX_NUM];
  for (int i = 0; i < num; ++i) {
    sorted_theta[i] = theta[i];
  }
  // bubble sort
  for (int i = 0; i < (num - 1); ++i) {
    for (int j = 0; j < (num - i - 1); ++j) {
      if (sorted_theta[j] < sorted_theta[j + 1]) {
        float temp = sorted_theta[j];
        sorted_theta[j] = sorted_theta[j + 1];
        sorted_theta[j + 1] = temp;
      }
    }
  }
  for (int i = 0; i < num; ++i) {
    tmp[i * 2] = tmp[i * 2] + center.x;
    tmp[i * 2 + 1] = tmp[i * 2 + 1] + center.y;
  }
  for (int i = 0; i < num; ++i) {
    float t = sorted_theta[i];
    int idx = -1;
    for (int j = 0; j < num; ++j) {
      if (theta[j] == t) {
        idx = j;
        break;
      }
    }
    pts[i * 2] = tmp[idx * 2];
    pts[i * 2 + 1] = tmp[idx * 2 + 1];
  }
}

PREFIX bool Exist(const Point& p, const float* pts) {
  for (int i = 0; i < MAX_NUM; ++i) {
    if (p.x == pts[i * 2] && p.y == pts[i * 2 + 1])
      return true;
  }
  return false;
}

PREFIX inline void Push(float* pts, const Point& p) {
  int idx = Length(pts, MAX_NUM);
  pts[idx * 2] = p.x;
  pts[idx * 2 + 1] = p.y;
}

PREFIX float QuadOverlaps(const float* q1, const float* q2) {
  // 0. init
  float pts[MAX_NUM * 2];
  for (int i = 0; i < MAX_NUM * 2; ++i)
    pts[i] = INVALID_VAL;
  // 1. get point inside of each quadrangle
  Point p, p1, p2, p3, p4;
  for (int i = 0; i < 4; ++i) {
    p1 = { q1[i * 2], q1[i * 2 + 1] };
    p2 = { q2[i * 2], q2[i * 2 + 1] };
    if (Contains(p1, q2) && !Exist(p1, pts))
      Push(pts, p1);
    if (Contains(p2, q1) && !Exist(p2, pts))
      Push(pts, p2);
  }
  // 2. get intersection by two edges
  float inter[4];
  int idx = 0;
  for (int i = 0; i < 4; ++i) {
    idx = (i + 1) > 3 ? 0 : (i + 1);
    p1 = { q1[i * 2], q1[i * 2 + 1] };
    p2 = { q1[idx * 2], q1[idx * 2 + 1] };
    for (int j = 0; j < 4; ++j) {
      idx = (j + 1) > 3 ? 0 : (j + 1);
      p3 = { q2[j * 2], q2[j * 2 + 1] };
      p4 = { q2[idx * 2], q2[idx * 2 + 1] };
      Intersect(p1, p2, p3, p4, inter);
      if (inter[0] != INVALID_VAL) {
        p = { inter[0], inter[1] };
        if (!Exist(p, pts))
          Push(pts, p);
      }
      if (inter[2] != INVALID_VAL) {
        p = { inter[2], inter[3] };
        if (!Exist(p, pts))
          Push(pts, p);
      }
    }
  }
  int num_pts = Length(pts, MAX_NUM);
  // if num of point less than 3, return 0
  if (3 > num_pts)
    return 0.f;
  // 3. sort points by clockwise or anticlockwise
  SortPoints(pts, num_pts);
  // 4. compute areas
  float ai = Area(pts, num_pts);
  float au = Area(q1, 4) + Area(q2, 4) - ai;
  return ai / max(au, 1.f);
}
