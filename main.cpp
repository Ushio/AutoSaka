#include "pr.hpp"
#include <iostream>
#include <memory>

#include "autodiff/reverse/var.hpp"

namespace saka
{
    class Val_;
    class Func_
    {
    public:
        std::shared_ptr<Val_> input;
        virtual ~Func_() {}
        virtual float forward(float x) const = 0;
        virtual float backward(float x, float dy) const = 0;
    };
    class Val_
    {
    public:
        float value;
        float derivative = 0.0f;
        std::shared_ptr<Func_> manufacturer;

        void backward(bool leaf)
        {
            if (!manufacturer)
            {
                return;
            }
            if (leaf)
            {
                derivative = 1.0f;
            }
            std::shared_ptr<Val_> input = manufacturer->input;
            input->derivative += manufacturer->backward(input->value, derivative);
            input->backward(false /*leaf*/);
        }
    };

    class Val
    {
    public:
        Val(float value):m_val(new Val_())
        {
            m_val->value = value;
        }
        void backward()
        {
            m_val->backward(true /*leaf*/);
        }
        float derivative() const {
            return m_val->derivative;
        }
        std::shared_ptr<Val_> m_val;
    };
    class Func
    {
    public:
        Func(std::shared_ptr<Func_> f):m_val(f){}
        Val forward(Val x)
        {
            float y = m_val->forward(x.m_val->value);
            Val r(y);
            m_val->input = x.m_val;
            r.m_val->manufacturer = m_val;
            return r;
        }
        std::shared_ptr<Func_> m_val;
    };

    class Square : public Func_
    {
    public:
        virtual float forward(float x) const
        {
            return x * x;
        }
        virtual float backward(float x, float dy) const
        {
            return dy * 2.0f * x;
        }
    };
    class Exp : public Func_
    {
    public:
        virtual float forward(float x) const
        {
            return std::exp(x);
        }
        virtual float backward(float x, float dy) const
        {
            return dy * std::exp(x);
        }
    };

    Val square(Val x)
    {
        Func f(std::shared_ptr<Func_>(new Square()));
        return f.forward(x);
    }
    Val exp(Val x)
    {
        Func f(std::shared_ptr<Func_>(new Exp()));
        return f.forward(x);
    }
}

using namespace autodiff;
//var f(var x, var y, var z)
//{
//    return (x + y + z) * exp(x * y * z);
//}
//var f(var x)
//{
//    return x * x;
//}
int main() {
    using namespace pr;

    {
        using namespace saka;
        Val val(1.2);
        Val r = exp(square(val));
        r.backward();

        float d = val.derivative();
        printf("%f\n", d);
    }

    {
        var x = 1.2;
        auto f = [](var x) { return exp( x * x ); };
        var y = f(x);
        auto [ux] = derivatives(y, wrt(x)); 
        printf("%f\n", ux);
    }

    //var x = 1.0;         // the input variable x
    //var y = 2.0;         // the input variable y
    //var z = 3.0;         // the input variable z
    //var u = f(x, y, z);  // the output variable u

    //auto [ux, uy, uz] = derivatives(u, wrt(x, y, z)); // evaluate the derivatives of u with respect to x, y, z

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
