#ifndef CURVE_CUH
#define CURVE_CUH

#include "vec3.cuh"
#include "hitable.cuh"
#include "common.cuh"
#include <vector>
struct CurvePoint{
    Vec3 v; // vertex
    Vec3 t; // tangent (unit)
};

class Curve : public Hitable {
public:
    int k;
    int n;
    int controls_num;
    int knot_num;
    int tpad_num;
    Vec3* controls;
    float* knot;
    float* tpad;
    float *s;
    float *ds;
    CurvePoint *eva;
	float max_r;
    constexpr static const float stride = 0.0001f;
    constexpr static const int eva_size = int(1 / stride) + 1;

    __device__ Curve(Vec3 *points, int n){
        this->controls_num = n;
        this->knot_num = 0;
        this->tpad_num = 0;
        this->controls = new Vec3[n];
        for(int i = 0; i < n; i++){
            this->controls[i] = points[i];
        }
        this->eva = new CurvePoint[eva_size];
    }

    __device__ virtual bool hit(const ray& r, float t_min, float t_max, HitRecord& rec) const {
        return false;
    }

    __device__ float solve(float x0){
        float t = 0.5f;
        float x;
        float dx;
        for (int i = 0; i < 10; i++){
            t = clamp(t);
            CurvePoint eval = evaluate(t);
            x = eval.v.x() - x0;
            dx = eval.t.x();
            if (abs(x) < eps) {
                return t;
            }
            t -= x / dx;
        }
        return -1;
    }

    __device__ void prepare(){
		max_r = 0;
        for(int i = 0; i < eva_size; i++){
            eva[i] = pre_evaluate(i * stride);
			if(max_r < eva[i].v.x()){
				max_r = eva[i].v.x();
			}
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
        Vec3 ret(0,0,0), retd(0,0,0);
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

    __device__ BezierCurve(Vec3 *points, int n) : Curve(points, n) {
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