#ifndef CURVE_CUH
#define CURVE_CUH

#include "vec3.cuh"
#include "hitable.cuh"
#include "common.cuh"

#include <thrust/device_vector.h>
#include <thrust/binary_search.h>

struct CurvePoint{
    vec3 v; // vertex
    vec3 t; // tangent (unit)
};

class Curve : public hitable {
public:
    thrust::device_vector<vec3> controls;
    thrust::device_vector<float> knot;
    int k;
    int n;
    thrust::device_vector<float> tpad;

    __device__ Curve(vec3 *points, int n){
        
    }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const {
        return false;
    }

    __device__ float solve(float y0){
        float t = 0.5f;
        float y;
        float dy;
        for (int i = 0; i < 10; i++){
            t = clamp(t);
            CurvePoint eval = evaluate(t);
            y = eval.v.y() - y0;
            dy = eval.t.y();
            if (abs(y) < eps) {
                return t;
            }
            t -= y / dy;
        }
        return -1;
    }

    
    __device__ virtual CurvePoint evaluate(float t){
        int bpos = getBPos(t);
        std::vector<float> s(k + 2, 0);
        s[k] = 1;
        std::vector<float> ds(k + 1, 1);
        for(int p = 1; p <= k; p++){
            for(int ii = k - p; ii <= k; ii++){
                int i = ii + bpos - k;
                float w1, w2, dw1, dw2;
                if (tpad[i + p] == tpad[i]){
                    w1 = t;
                    dw1 = 1;
                }else{
                    w1 = (t - tpad[i]) / (tpad[i + p] - tpad[i]);
                    dw1 = 1 / (tpad[i+p] - tpad[i]);
                }
                if (tpad[i + p + 1] == tpad[i + 1]){
                    w2 = 1 - t;
                    dw2 = -1;
                } else {
                    w2 = (tpad[i + p + 1] - t) / (tpad[i + p + 1] - tpad[i + 1]);
                    dw2 = - 1 / (tpad[i + p + 1] - tpad[i + 1]);
                }
                if (p == k){
                    ds[ii] = (dw1 * s[ii] + dw2 * s[ii + 1]) * p;
                }
                s[ii] = w1 * s[ii] + w2 * s[ii + 1];
            }
        }
        vec3 ret(0,0,0), retd(0,0,0);
        int lsk = bpos - k;
        int rsk = n - bpos;
        int le = max(0, lsk);
        int ma = min(k, k + rsk) + le;
        for(int i = le; i <= ma; i++){
            ret += this->controls[i] * s[i - lsk];
            retd += this->controls[i] * ds[i - lsk];
        }
        return CurvePoint{ret, retd};
    }

    __device__ virtual void discretize(int resolution, std::vector<CurvePoint>& data) = 0;

    __device__ virtual void caculKnot() = 0;

    __device__ int getBPos(float mu){
        int pos;
        if(mu == 0){
            pos = thrust::upper_bound(knot.data(), knot.data() + knot.size(), mu) - knot.data() - 1;
        }else{
            pos = thrust::lower_bound(knot.data(), knot.data() + knot.size(), mu) - knot.data() - 1;
            if(pos < 0){
                pos = 0;
            }
        }
        return pos;
    }

};


class BezierCurve : public Curve {
public:

    __device__ BezierCurve(vec3 *points, int n) : Curve(points, n) {
        if (n < 4 || n % 3 != 1) {
            printf("Number of control points of BezierCurve must be 3n+1!\n");
            exit(0);
        }
        this->k = n - 1;
        this->n = n - 1;
        caculKnot();
    }

    __device__ void discretize(int resolution, std::vector<CurvePoint>& data) override {
        data.clear();
        // DONE (PA3): fill in data vector
        for(int i = 0; i <= resolution; i++){
            float t = float(i) / resolution;
            data.push_back(evaluate(t));
        }
    }

    __device__ void caculKnot(){
        for(int i = 0; i < k + 1; i++){
            knot.push_back(0);
            tpad.push_back(0);
        }
        for(int i = 0; i < k + 1; i++){
            knot.push_back(1);
            tpad.push_back(1);
        }
        for(int i = 0;i < k; i++){
            tpad.push_back(1);
        }
    }


protected:
};

#endif