
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pangolin/handler/handler.h>
#include <pangolin/gl/gldraw.h>

#include <pangolin/utils/posix/condition_variable.h>
#include <pangolin/utils/posix/shared_memory_buffer.h>
#include <pangolin/utils/timer.h>

#include <cmath>
#include <memory>

int pangolinTest()
{
    pangolin::CreateWindowAndBind("Main",640,480);
    glEnable(GL_DEPTH_TEST);

    // Define Projection and initial ModelView matrix
    pangolin::OpenGlRenderState s_cam(
        pangolin::ProjectionMatrix(640,480,420,420,320,240,0.2,100),
        pangolin::ModelViewLookAt(-2,2,-2, 0,0,0, pangolin::AxisY)
    );

    // Create Interactive View in window
    pangolin::Handler3D handler(s_cam);
    pangolin::View& d_cam = pangolin::CreateDisplay()
            .SetBounds(0.0, 1.0, 0.0, 1.0, -640.0f/480.0f)
            .SetHandler(&handler);

    while( !pangolin::ShouldQuit() )
    {
        // Clear screen and activate view to render into
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        d_cam.Activate(s_cam);

        // Render OpenGL Cube
        pangolin::glDrawColouredCube();

        // Swap frames and Process Events
        pangolin::FinishFrame();
    }

    return 0;
}

unsigned char generate_value(double t)
{
  // 10s sinusoid
  const double d = std::sin(t * 10.0 / M_PI) * 128.0 + 128.0;
  return static_cast<unsigned char>(d);
}

int sharedMemoryTest(/*int argc, char *argv[]*/)
{
  std::string shmem_name = "/example";

  std::shared_ptr<pangolin::SharedMemoryBufferInterface> shmem_buffer =
    pangolin::create_named_shared_memory_buffer(shmem_name, 640 * 480);
  if (!shmem_buffer) {
    perror("Unable to create shared memory buffer");
    exit(1);
  }

  std::string cond_name = shmem_name + "_cond";
  std::shared_ptr<pangolin::ConditionVariableInterface> buffer_full =
    pangolin::create_named_condition_variable(cond_name);

  // Sit in a loop and write gray values based on some timing pattern.
  while (true) {
    shmem_buffer->lock();
    unsigned char *ptr = shmem_buffer->ptr();
    unsigned char value = generate_value(std::chrono::system_clock::now().time_since_epoch().count());

    for (int i = 0; i < 640*480; ++i) {
      ptr[i] = value;
    }

    shmem_buffer->unlock();
    buffer_full->signal();

    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
}