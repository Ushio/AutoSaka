#include "catch_amalgamated.hpp"
#include "pr.hpp"
#include "autodiff/reverse/var.hpp"
#include "saka.h"

#include <functional>

using namespace autodiff;
using namespace saka;

template <class T>
T simple_1( T x0 )
{
    return exp(x0 * x0);
}
template <class T>
T four_arithmetic_ops(T x0)
{
    return (x0 * x0 * T(2.0f) - x0) / x0 + x0;
}

bool almostEqual(float val, float val_ref, int toleranceScale )
{
    return fabs(val - val_ref) < fabs(val_ref) * FLT_EPSILON * toleranceScale;
}

TEST_CASE("Simples", "") {

    struct unary_function
    {

        std::function<float(float)> f_basic;
        std::function<DVal<1>(DVal<1>)> f_saka;
        std::function<var(var)> f_var;
        
        float input_lower, input_upper;

        float operator()(float x) const { return f_basic(x); }
        DVal<1> operator()(DVal<1> x) const { return f_saka(x); }
        var operator()(var x) const { return f_var(x); }
    };

    std::vector<unary_function> fs = {
        {simple_1<float>, simple_1<DVal<1>>, simple_1<var>, -3.0f, 3.0f},
        {four_arithmetic_ops<float>, four_arithmetic_ops<DVal<1>>, four_arithmetic_ops<var>, -4.0f, -0.5f},
    };

    pr::PCG rng;

    for (auto f : fs)
    {
        for (int i = 0; i < 100; i++)
        {
            float x0_input = glm::mix(f.input_lower, f.input_upper, rng.uniformf());
            float y_ref = f(x0_input);

            float dy_x0;
            float dy_x0_ref;
            {
            
                DVal<1> x0(x0_input); x0.requireDerivative(0);
                DVal<1> y = f(x0);

                REQUIRE(almostEqual(y.value, y_ref, 32));
                dy_x0 = y.dvalues[0];
            }

            {
                var x0 = x0_input;
                var y = f(x0);
                auto [dy] = derivatives(y, wrt(x0));
                dy_x0_ref = dy;
            }
            REQUIRE(almostEqual(dy_x0, dy_x0_ref, 32));
        }
    }
}

