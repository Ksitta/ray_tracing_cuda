#ifndef CURVE_CUH
#define CURVE_CUH

#include "vec3.cuh"
#include "hitable.cuh"
#include "common.cuh"
#include <vector>
struct CurvePoint{
    vec3 v; // vertex
    vec3 t; // tangent (unit)
};

class Curve : public hitable {
public:
    int k;
    int n;
    int controls_num;
    int knot_num;
    int tpad_num;
    vec3* controls;
    float* knot;
    float* tpad;
    float *s;
    float *ds;
    CurvePoint *eva;
    constexpr static const float stride = 0.00001f;
    constexpr static const int eva_size = int(1 / stride) + 1;

    __device__ Curve(vec3 *points, int n){
        this->controls_num = n;
        this->knot_num = 0;
        this->tpad_num = 0;
        this->controls = new vec3[n];
        for(int i = 0; i < n; i++){
            this->controls[i] = points[i];
        }
        this->eva = new CurvePoint[eva_size];
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

    __device__ void prepare(){
        for(int i = 0; i < eva_size; i++){
            eva[i] = pre_evaluate(i * stride);
        }
    }

    __device__ CurvePoint evaluate(float t){
        return eva[int(t / stride)];
    }
    
    __device__ CurvePoint pre_evaluate(float t){
        int bpos = getBPos(t);
        
        for (int i = 0; i < k + 2; i++){
            s[i] = 0;
        }
        s[k] = 1;

        for(int i = 0; i < k + 1; i++){
            ds[i] = 1;
        }
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

    __device__ virtual void caculKnot() = 0;

    __device__ int upper_bound(float *data, int n, float target){
        int l = 0, r = n - 1;
        while(l < r){
            int mid = (l + r) / 2;
            if(data[mid] <= target){
                l = mid + 1;
            }else{
                r = mid;
            }
        }
        return l;
    }

    __device__ int lower_bound(float *data, int n, float target){
        int l = 0, r = n - 1;
        while(l < r){
            int mid = (l + r) / 2;
            if(data[mid] < target){
                l = mid + 1;
            }else{
                r = mid;
            }
        }
        return l;
    }

    __device__ int getBPos(float mu){
        int pos;
        if(mu == 0){
            pos = upper_bound(knot, knot_num, mu) - 1;
        }else{
            pos = lower_bound(knot, knot_num, mu) - 1;
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
            return;
        }
        this->k = n - 1;
        this->n = n - 1;
        this->knot = new float[2 * (k + 1)];
        this->tpad = new float[2 * (k + 1) + k];
        s = new float[k + 2];
        ds = new float[k + 1];
        caculKnot();
        prepare();
    }

    __device__ void caculKnot(){
        for(int i = 0; i < k + 1; i++){
            knot[knot_num++] = 0;
            tpad[tpad_num++] = 0;
        }
        for(int i = 0; i < k + 1; i++){
            knot[knot_num++] = 1;
            tpad[tpad_num++] = 1;
        }
        for(int i = 0;i < k; i++){
            tpad[tpad_num++] = 1;
        }
    }


protected:
};

#endif