#include "catch_amalgamated.hpp"
#include "pr.hpp"
#include <autodiff/forward/dual.hpp>
#include "saka.h"

#include <functional>

using namespace autodiff;
using namespace saka;

dual simple_0_ref(dual x)
{
    return x * x;
}
dval simple_0(dval x)
{
    return x * x;
}

TEST_CASE("simple_0", "") {
    pr::PCG rng;

    for (int i = 0; i < 1000; i++)
    {
        dual x_ref = rng.uniformf();
        double dudx = derivative(simple_0_ref, wrt(x_ref), at(x_ref));

        dval x = x_ref.val; x.requires_grad();
        dval u = simple_0(x);

        REQUIRE( fabsf( dudx - u.g ) < 1.0e-5f );
    }
}

dual simple_1_ref(dual x)
{
    return exp(x * x);
}
dval simple_1(dval x)
{
    return exp(x * x);
}

TEST_CASE("simple_1", "") {
    pr::PCG rng;

    for (int i = 0; i < 1000; i++)
    {
        dual x_ref = rng.uniformf();
        double dudx = derivative(simple_1_ref, wrt(x_ref), at(x_ref));

        dval x = x_ref.val; x.requires_grad();
        dval u = simple_1(x);

        REQUIRE(fabsf(dudx - u.g) < 1.0e-5f);
    }
}

dual complex_0_ref(dual x, dual y, dual z)
{
    return 1 + x + y + z + x * y + y * z + x * z + x * y * z + exp(x / y + y / z);
}
dval complex_0(dval x, dval y, dval z)
{
    return 1 + x + y + z + x * y + y * z + x * z + x * y * z + exp(x / y + y / z);
}

TEST_CASE("complex_0", "") {
    pr::PCG rng;

    for (int i = 0; i < 1000; i++)
    {
        dual x_ref = 1.0f + rng.uniformf();
        dual y_ref = 1.0f + rng.uniformf();
        dual z_ref = 1.0f + rng.uniformf();
        double dudx = derivative(complex_0_ref, wrt(x_ref), at(x_ref, y_ref, z_ref));

        dval x = x_ref.val; x.requires_grad();
        dval y = y_ref.val;
        dval z = z_ref.val;
        dval u = complex_0(x, y, z);

        REQUIRE(fabsf(dudx - u.g) < 1.0e-5f);
    }
}
dual complex_1_ref(dual x, dual y, dual z)
{
    return (x + y + z) * exp(x * y * z);
}
dval complex_1(dval x, dval y, dval z)
{
    return (x + y + z) * exp(x * y * z);
}

TEST_CASE("complex_1", "") {
    pr::PCG rng;

    for (int i = 0; i < 1000; i++)
    {
        dual x_ref = -1.0f + 2.0f * rng.uniformf();
        dual y_ref = -1.0f + 2.0f * rng.uniformf();
        dual z_ref = -1.0f + 2.0f * rng.uniformf();
        double dudx = derivative(complex_1_ref, wrt(x_ref), at(x_ref, y_ref, z_ref));

        dval x = x_ref.val; x.requires_grad();
        dval y = y_ref.val;
        dval z = z_ref.val;
        dval u = complex_1(x, y, z);

        REQUIRE(fabsf(dudx - u.g) < 1.0e-5f);
    }
}