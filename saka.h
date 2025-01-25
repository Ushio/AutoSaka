#pragma once

namespace saka
{
    // -- forward module -- 
    template <int NDerivatives>
    class DVal
    {
    public:
        DVal() {}
        DVal(float v) :value(v)
        {
            for (int i = 0; i < NDerivatives; i++)
            {
                dvalues[i] = 0.0f;
            }
        }
        void requireDerivative(int indexOfVal)
        {
            dvalues[indexOfVal] = 1.0f;
        }
        float value;
        float dvalues[NDerivatives];
    };

    template <int NDerivatives, class F, class DF>
    inline DVal<NDerivatives> apply(DVal<NDerivatives> x, F f, DF df)
    {
        DVal<NDerivatives> r;
        r.value = f(x.value);
        for (int i = 0; i < NDerivatives; i++)
        {
            r.dvalues[i] = x.dvalues[i] * df(x.value);
        }
        return r;
    }
    template <int NDerivatives, class F, class DF0, class DF1>
    inline DVal<NDerivatives> apply(DVal<NDerivatives> x0, DVal<NDerivatives> x1, F f, DF0 df0, DF1 df1)
    {
        DVal<NDerivatives> r;
        r.value = f(x0.value, x1.value);
        for (int i = 0; i < NDerivatives; i++)
        {
            r.dvalues[i] = x0.dvalues[i] * df0(x0.value, x1.value) + x1.dvalues[i] * df1(x0.value, x1.value);
        }
        return r;
    }

    template <int NDerivatives>
    inline DVal<NDerivatives> exp(DVal<NDerivatives> x)
    {
        return apply(x, [](float x) { return std::expf(x); }, [](float x) { return std::expf(x); });
    }

    template <int NDerivatives>
    inline DVal<NDerivatives> operator+(DVal<NDerivatives> x0, DVal<NDerivatives> x1)
    {
        return apply(x0, x1,
            [](float x0, float x1) { return x0 + x1; },
            [](float x0, float x1) { return 1.0f; },
            [](float x0, float x1) { return 1.0f; });
    }
    template <int NDerivatives>
    inline DVal<NDerivatives> operator-(DVal<NDerivatives> x0, DVal<NDerivatives> x1)
    {
        return apply(x0, x1,
            [](float x0, float x1) { return x0 - x1; },
            [](float x0, float x1) { return 1.0f; },
            [](float x0, float x1) { return -1.0f; });
    }
    template <int NDerivatives>
    inline DVal<NDerivatives> operator*(DVal<NDerivatives> x0, DVal<NDerivatives> x1)
    {
        return apply(x0, x1,
            [](float x0, float x1) { return x0 * x1; },
            [](float x0, float x1) { return x1; },
            [](float x0, float x1) { return x0; });
    }
    template <int NDerivatives>
    inline DVal<NDerivatives> operator/(DVal<NDerivatives> x0, DVal<NDerivatives> x1)
    {
        return apply(x0, x1,
            [](float x0, float x1) { return x0 / x1; },
            [](float x0, float x1) { return 1.0f / x1; },
            [](float x0, float x1) { return -x0 / (x1 * x1); });
    }
}
