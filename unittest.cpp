#include "catch_amalgamated.hpp"
#include "pr.hpp"
#include "autodiff/reverse/var.hpp"
#include "saka.h"

using namespace autodiff;
using namespace saka;

template <class T>
T simple_1( T x0 )
{
    return exp(x0 * x0);
}

bool almostEqual(float val, float val_ref, int toleranceScale )
{
    return fabs(val - val_ref) < fabs(val_ref) * FLT_EPSILON * toleranceScale;
}

TEST_CASE("Simples", "") {
    pr::PCG rng;

    for (int i = 0; i < 100; i++)
    {
        float x0_input = rng.uniformf();
        float y_ref = simple_1(x0_input);

        float dy_x0;
        float dy_x0_ref;
        {
            
            DVal<1> x0(x0_input); x0.requireDerivative(0);
            DVal<1> y = simple_1(x0);

            REQUIRE(almostEqual(y.value, y_ref, 4));
            dy_x0 = y.dvalues[0];
        }

        {
            var x0 = x0_input;
            var y = simple_1(x0);
            auto [dy] = derivatives(y, wrt(x0));
            dy_x0_ref = dy;
        }
        REQUIRE(almostEqual(dy_x0, dy_x0_ref, 4));
    }
}

template <class T>
T four_arithmetic_ops(T x0)
{
    return (x0 * x0 * T(2.0f) - x0) / x0 + x0;
}

TEST_CASE("four arithmetic ops", "") {
    pr::PCG rng;

    for (int i = 0; i < 100; i++)
    {
        float x0_input = glm::mix(-10.0f, -1.0f, rng.uniformf());
        float y_ref = four_arithmetic_ops(x0_input);

        float dy_x0;
        float dy_x0_ref;
        {

            DVal<1> x0(x0_input); x0.requireDerivative(0);
            DVal<1> y = four_arithmetic_ops(x0);

            bool b = almostEqual(y.value, y_ref, 4);
            REQUIRE(almostEqual(y.value, y_ref, 4));
            dy_x0 = y.dvalues[0];
        }

        {
            var x0 = x0_input;
            var y = four_arithmetic_ops(x0);
            auto [dy] = derivatives(y, wrt(x0));
            dy_x0_ref = dy;
        }
        REQUIRE(almostEqual(dy_x0, dy_x0_ref, 4));
    }
}
