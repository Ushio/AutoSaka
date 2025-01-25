#include "pr.hpp"
#include <iostream>

#include "saka.h"
#include "autodiff/reverse/var.hpp"

using namespace autodiff;

int main() {
    using namespace pr;

    {
        using namespace saka;
        DVal<1> val(1.2);
        val.requireDerivative(0);
        DVal<1> r = exp(val * val);

        printf("%f\n", r.dvalues[0]);
    }
    {
        var x = 1.2;
        auto f = [](var x) { return exp( x * x ); };
        var y = f(x);
        auto [ux] = derivatives(y, wrt(x)); 
        printf("%f\n", ux);
    }

    printf("--\n");

    {
        using namespace saka;
        DVal<1> val(1.4);
        val.requireDerivative(0);
        DVal<1> r = val * val + val * val;

        printf("%f\n", r.dvalues[0]);
        //printf("%s\n", r.dotLang().c_str());
    }
    {
        var x = 1.4;
        auto f = [](var x) { return x * x + x * x; };
        var y = f(x);
        auto [ux] = derivatives(y, wrt(x));
        printf("%f\n", ux);
    }

    printf("--\n");

    // A complex graph
    //{
    //    using namespace saka;
    //    ValRef x(1.4f);
    //    ValRef a = square(x);
    //    //ValRef a = x;
    //    ValRef b = exp(a);
    //    ValRef c = square(a);
    //    ValRef y = b / c;
    //    
    //    y.backward();

    //    float d = x.derivative();
    //    printf("%f\n", d);

    //    printf("%s\n", y.dotLang().c_str());
    //}
    {
        using namespace saka;
        DVal<1> x(1.4f);
        x.requireDerivative(0);
        DVal<1> a = x * x;
        //ValRef a = x;
        DVal<1> b = exp(a);
        DVal<1> c = a * a;
        DVal<1> y = b / c;

        float d = y.dvalues[0];
        printf("%f\n", d);
    }

    {
        var x = 1.4f;
        auto f = [](var x) {
            auto a = x * x;
            //auto a = x;
            auto b = exp(a);
            auto c = a * a;
            return b / c;
        };
        var y = f(x);
        auto [ux] = derivatives(y, wrt(x));
        printf("%f\n", ux);
    }

    //{
    //    var x = 1.4f;
    //    auto f = [](var x) {
    //        auto a = x * x;
    //        //auto a = x;
    //        auto b = exp(a);
    //        auto c = a * a;
    //        return saka::Pair<var>(b * c, b + c);
    //    };
    //    saka::Pair<var> y = f(x);
    //    auto [dydx0] = derivatives(y.lhs, wrt(x));
    //    auto [dydx1] = derivatives(y.rhs, wrt(x));
    //    printf("%f\n", dydx0);
    //    printf("%f\n", dydx1);
    //}

    //var x = 1.0;         // the input variable x
    //var y = 2.0;         // the input variable y
    //var z = 3.0;         // the input variable z
    //var u = f(x, y, z);  // the output variable u

    //auto [ux, uy, uz] = derivatives(u, wrt(x, y, z)); // evaluate the derivatives of u with respect to x, y, z

    Xoshiro128StarStar rng;
    for (int i = 0; i < 10; i++)
    {
        printf("%f\n", rng.uniformf());
    }

    Config config;
    config.ScreenWidth = 1920;
    config.ScreenHeight = 1080;
    config.SwapInterval = 1;
    Initialize(config);

    Camera3D camera;
    camera.origin = { 4, 4, 4 };
    camera.lookat = { 0, 0, 0 };
    camera.zUp = true;

    double e = GetElapsedTime();

    while (pr::NextFrame() == false) {
        if (IsImGuiUsingMouse() == false) {
            UpdateCameraBlenderLike(&camera);
        }

        ClearBackground(0.1f, 0.1f, 0.1f, 1);

        BeginCamera(camera);

        PushGraphicState();

        //DrawGrid(GridAxis::XY, 1.0f, 10, { 128, 128, 128 });
        //DrawXYZAxis(1.0f);

        PopGraphicState();
        EndCamera();

        BeginImGui();

        ImGui::SetNextWindowSize({ 500, 800 }, ImGuiCond_Once);
        ImGui::Begin("Panel");
        ImGui::Text("fps = %f", GetFrameRate());

        ImGui::End();

        EndImGui();
    }

    pr::CleanUp();
}
