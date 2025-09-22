#pragma once

namespace saka
{
    class dval
    {
    public:
        dval(): v(0.0f), g(0.0f) {}
        dval(float x) :v(x), g(0.0f) {}

        void requires_grad()
        {
            g = 1.0f;
        }

        float v;
        float g;
    };

    namespace details
    {
        template <class F, class dFdx>
        inline dval unary(dval x, F f, dFdx dfdx)
        {
            dval u;
            u.v = f(x.v);
            u.g = x.g * dfdx(x.v);
            return u;
        }

        template <class F, class dFdx, class dFdy>
        inline dval binary(dval x, dval y, F f, dFdx dfdx, dFdy dfdy)
        {
            dval u;
            u.v = f(x.v, y.v);
            u.g = x.g * dfdx(x.v, y.v) + y.g * dfdy(x.v, y.v);
            return u;
        }
    }
    inline dval operator+(dval x, dval y)
    {
        return details::binary(x, y,
            [](float x, float y) { return x + y; },
            [](float x, float y) { return 1.0f; }, // df/dx
            [](float x, float y) { return 1.0f; }  // df/dy
        );
    }
    inline dval operator-(dval x, dval y)
    {
        return details::binary(x, y,
            [](float x, float y) { return x - y; },
            [](float x, float y) { return +1.0f; },
            [](float x, float y) { return -1.0f; });
    }
    inline dval operator*(dval x, dval y)
    {
        return details::binary(x, y,
            [](float x, float y) { return x * y; },
            [](float x, float y) { return y; }, // df/dx
            [](float x, float y) { return x; }  // df/dy
        );
    }
    inline dval operator/(dval x, dval y)
    {
        return details::binary(x, y,
            [](float x, float y) { return x / y; },
            [](float x, float y) { return 1.0f / y; },
            [](float x, float y) { return -x / (y * y); });
    }
    inline dval exp(dval x)
    {
        return details::unary(x, 
            [](float x) { return expf(x); }, 
            [](float x) { return expf(x); } // df/dx
        );
    }
    inline dval sqrt(dval x)
    {
        return details::unary(x, 
            [](float x) { return sqrtf(x); }, 
            [](float x) { return 0.5f / sqrtf(x); } // df/dx
        );
    }


}
