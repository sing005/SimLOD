
#include <cstdlib>
#include <filesystem>

#include "GLRenderer.h"
#include "Runtime.h"

namespace fs = std::filesystem;

auto _controls = make_shared<OrbitControls>();


static void APIENTRY debugCallback(GLenum source, GLenum type, GLuint id, GLenum severity, GLsizei length, const GLchar* message, const void* userParam) {

	if (
		severity == GL_DEBUG_SEVERITY_NOTIFICATION 
		|| severity == GL_DEBUG_SEVERITY_LOW 
		|| severity == GL_DEBUG_SEVERITY_MEDIUM
		) {
		return;
	}

	cout << "OPENGL DEBUG CALLBACK: " << message << endl;
}

static void report(const string& message){
	if(GLRenderer::log) GLRenderer::log(message);
	else fprintf(stderr, "%s\n", message.c_str());
}

// GLFW errors were written to stderr, which the run log does not see. The
// first 20 are logged, in case one repeats every frame.
void error_callback(int error, const char* description){
	static int numReported = 0;
	if(numReported >= 20) return;
	numReported++;
	report("GLFW error " + to_string(error) + ": " + (description ? description : "") + (numReported == 20 ? " (further GLFW errors are not logged)" : ""));
}

static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods){

	if(GLRenderer::inputLocked) return;

	// GLFW passes GLFW_KEY_UNKNOWN (-1) for keys it has no token for, such as
	// media keys (glfw3.h, glfwSetKeyCallback); it was written to keyStates[-1]
	if(key < 0 || key >= int(Runtime::keyStates.size())) return;

	// Esc closes the viewer, except while a text field or slider has the
	// keyboard (there Esc cancels the edit) or a popup is open, such as a
	// colour picker: ImGui 1.81 counts neither popups nor non-modal windows in
	// WantCaptureKeyboard, so Esc pressed to dismiss a picker closed the viewer
	// and lost unsaved markers and the loaded cloud.
	bool popupOpen = ImGui::IsPopupOpen("", ImGuiPopupFlags_AnyPopup);   // reads the popup stack only
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS && !ImGui::GetIO().WantCaptureKeyboard && !popupOpen) {
		glfwSetWindowShouldClose(window, GLFW_TRUE);
	}

	Runtime::keyStates[key] = action;
}

static void cursor_position_callback(GLFWwindow* window, double xpos, double ypos){
	ImGuiIO& io = ImGui::GetIO();
	if(io.WantCaptureMouse || GLRenderer::inputLocked){
		return;
	}
	
	Runtime::mousePosition = {xpos, ypos};

	_controls->onMouseMove(xpos, ypos);
}

void scroll_callback(GLFWwindow* window, double xoffset, double yoffset){
	ImGuiIO& io = ImGui::GetIO();
	if(io.WantCaptureMouse || GLRenderer::inputLocked){
		return;
	}

	_controls->onMouseScroll(xoffset, yoffset);
}


void mouse_button_callback(GLFWwindow* window, int button, int action, int mods){

	// cout << "start button: " << button << ", action: " << action << ", mods: " << mods << endl;

	ImGuiIO& io = ImGui::GetIO();
	if(io.WantCaptureMouse || GLRenderer::inputLocked){
		return;
	}

	// cout << "end button: " << button << ", action: " << action << ", mods: " << mods << endl;


	if(action == 1){
		Runtime::mouseButtons = Runtime::mouseButtons | (1 << button);
	}else if(action == 0){
		uint32_t mask = ~(1 << button);
		Runtime::mouseButtons = Runtime::mouseButtons & mask;
	}

	_controls->onMouseButton(button, action, mods);
}

GLRenderer::GLRenderer(){
	this->controls = _controls;
	camera = make_shared<Camera>();

	init();

	view.framebuffer = this->createFramebuffer(128, 128);
}

void GLRenderer::init(){
	glfwSetErrorCallback(error_callback);

	if (!glfwInit()) {
		report("GLFW could not start, so no window can open");
		exit(EXIT_FAILURE);
	}

	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 6);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_DECORATED, true);
	glfwWindowHint(GLFW_RESIZABLE, true);
	glfwWindowHint(GLFW_VISIBLE, !startHidden);
	// Nothing is drawn into the window with a depth or stencil test: it gets
	// the frame (a blit, colour only) and the UI, which ImGui draws with both
	// tests off. GLFW's defaults asked for 24 depth and 8 stencil bits.
	glfwWindowHint(GLFW_DEPTH_BITS, 0);
	glfwWindowHint(GLFW_STENCIL_BITS, 0);

	int numMonitors;
	GLFWmonitor** monitors = glfwGetMonitors(&numMonitors);

	cout << "<create windows>" << endl;
	if (numMonitors > 1) {
		window = glfwCreateWindow(1920, 1080, "SimLOD", nullptr, nullptr);

		if (!window) {
			report("no OpenGL 4.6 core-profile window could be created: the graphics driver may not support OpenGL 4.6");
			glfwTerminate();
			exit(EXIT_FAILURE);
		}

		// SECOND MONITOR
		//int xpos;
		//int ypos;
		// glfwGetMonitorPos(monitors[1], &xpos, &ypos);
		// glfwSetWindowPos(window, xpos, ypos);

		// FIRST
		// glfwSetWindowPos(window, 0, 0);
	} else {
		const GLFWvidmode * mode = numMonitors > 0 ? glfwGetVideoMode(monitors[0]) : nullptr;   // no monitor: monitors is null

		window = glfwCreateWindow(mode ? mode->width - 100 : 1920, mode ? mode->height - 100 : 1080, "SimLOD", nullptr, nullptr);

		if (!window) {
			report("no OpenGL 4.6 core-profile window could be created: the graphics driver may not support OpenGL 4.6");
			glfwTerminate();
			exit(EXIT_FAILURE);
		}

		glfwSetWindowPos(window, 50, 50);
	}

	cout << "<set input callbacks>" << endl;
	glfwSetKeyCallback(window, key_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
	glfwSetScrollCallback(window, scroll_callback);

	static GLRenderer* ref = this;
	glfwSetDropCallback(window, [](GLFWwindow*, int count, const char **paths){

		vector<string> files;
		for(int i = 0; i < count; i++){
			string file = paths[i];
			files.push_back(file);
		}

		for(auto &listener : ref->fileDropListeners){
			listener(files);
		}
	});

	glfwMakeContextCurrent(window);
	glfwSwapInterval(0);

	GLenum err = glewInit();
	if (GLEW_OK != err) {
		// start-up used to go on, to fail at the first OpenGL call
		report(string("GLEW could not load the OpenGL functions: ") + (const char*)glewGetErrorString(err));
		exit(EXIT_FAILURE);
	}

	cout << "<glewInit done> " << "(" << now() << ")" << endl;

	glEnable(GL_DEBUG_OUTPUT);
	glEnable(GL_DEBUG_OUTPUT_SYNCHRONOUS);
	glDebugMessageControl(GL_DONT_CARE, GL_DONT_CARE, GL_DEBUG_SEVERITY_HIGH, 0, NULL, GL_TRUE);
	glDebugMessageCallback(debugCallback, NULL);

	{ // SETUP IMGUI
		IMGUI_CHECKVERSION();
		ImGui::CreateContext();
		ImPlot::CreateContext();
		ImGui_ImplGlfw_InitForOpenGL(window, true);
		ImGui_ImplOpenGL3_Init("#version 450");
		ImGui::StyleColorsDark();
	}
}

shared_ptr<Texture> GLRenderer::createTexture(int width, int height, GLuint colorType) {

	auto texture = Texture::create(width, height, colorType, this);

	return texture;
}

shared_ptr<Framebuffer> GLRenderer::createFramebuffer(int width, int height) {

	auto framebuffer = Framebuffer::create(this);

	GLenum status = glCheckNamedFramebufferStatus(framebuffer->handle, GL_FRAMEBUFFER);

	if (status != GL_FRAMEBUFFER_COMPLETE) {
		cout << "framebuffer incomplete" << endl;
	}

	framebuffer->setSize(width, height);

	return framebuffer;
}

void GLRenderer::loop(function<void(void)> update, function<void(void)> render){

	int fpsCounter = 0;
	double start = now();
	double tPrevious = start;
	double tPreviousFPSMeasure = start;

	while (!glfwWindowShouldClose(window)){

		// TIMING
		double timeSinceLastFrame;
		{
			double tCurrent = now();
			timeSinceLastFrame = tCurrent - tPrevious;
			tPrevious = tCurrent;

			double timeSinceLastFPSMeasure = tCurrent - tPreviousFPSMeasure;

			if(timeSinceLastFPSMeasure >= 1.0){
				this->fps = double(fpsCounter) / timeSinceLastFPSMeasure;

				tPreviousFPSMeasure = tCurrent;
				fpsCounter = 0;
			}
		}

		// Minimised: wait for events, up to 0.1 s, instead of drawing frames
		// no one sees, unless the application has work that needs frames.
		// The loop used to run flat out.
		if(glfwGetWindowAttrib(window, GLFW_ICONIFIED) && !(workPending && workPending())){
			glfwWaitEventsTimeout(0.1);
			continue;
		}

		// WINDOW
		int width, height;
		glfwGetWindowSize(window, &width, &height);
		this->windowWidth = width;
		this->windowHeight = height;
		// a fixed render size, if one is set, independent of the window
		if(renderWidth > 0 && renderHeight > 0 && width > 0 && height > 0){
			width = renderWidth;
			height = renderHeight;
		}
		camera->setSize(width, height);
		this->width = width;
		this->height = height;

		EventQueue::instance->process();

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, this->width, this->height);

		glBindFramebuffer(GL_FRAMEBUFFER, view.framebuffer->handle);
		glViewport(0, 0, this->width, this->height);


		{ 
			controls->update();

			camera->world = controls->world;
			camera->position = camera->world * dvec4(0.0, 0.0, 0.0, 1.0);
		}

		ImGui_ImplOpenGL3_NewFrame();
		ImGui_ImplGlfw_NewFrame();
		ImGui::NewFrame();

		{ // UPDATE & RENDER
			camera->update();
			update();
			camera->update();

			render();
		}

		// IMGUI
		{
			ImGui::SetNextWindowPos(ImVec2(10, 0));
			ImGui::SetNextWindowSize(ImVec2(490, 15));
			ImGui::Begin("Test", nullptr,
				ImGuiWindowFlags_NoTitleBar 
				| ImGuiWindowFlags_NoScrollbar
				| ImGuiWindowFlags_NoMove
				| ImGuiWindowFlags_NoResize
				| ImGuiWindowFlags_NoBackground
			);

			bool toggleGUI = ImGui::Button("Toggle GUI");
			if(toggleGUI){
				Runtime::showGUI = !Runtime::showGUI;
			}
			ImGui::End();
		}

		auto windowSize_perf = ImVec2(490, 260);

		if(Runtime::showGUI && Runtime::showPerfGraph)
		// if(false)
		{ // RENDER IMGUI PERFORMANCE WINDOW

			stringstream ssFPS; 
			ssFPS << this->fps;
			string strFps = ssFPS.str();

			// movable + collapse-to-title (apply_windows.ps1)
			static bool   perfCollapsed = false;
			static ImVec2 perfSaved     = ImVec2(0, 0);
			ImGui::SetNextWindowPos(ImVec2(10, 30), ImGuiCond_FirstUseEver);
			if(perfCollapsed){
				float w = ImGui::CalcTextSize("Performance").x
						+ ImGui::GetStyle().WindowPadding.x * 2.0f
						+ ImGui::GetFrameHeight() + ImGui::GetFontSize();
				ImGui::SetNextWindowSize(ImVec2(w, 0.0f), ImGuiCond_Always);
			}else{
				ImGui::SetNextWindowSize(perfSaved.x > 0.0f ? perfSaved : windowSize_perf,
										 ImGuiCond_FirstUseEver);
			}

			ImGui::Begin("Performance");
			{
				bool nc = ImGui::IsWindowCollapsed();
				if(!nc && perfCollapsed && perfSaved.x > 0.0f){
					// just expanded: the narrow collapsed width was applied above
					// for this frame, so restore the saved size now and do not
					// record the narrow one over it
					ImGui::SetWindowSize(perfSaved, ImGuiCond_Always);
				}else if(!nc){
					ImVec2 s = ImGui::GetWindowSize();
					if(s.x > 0.0f) perfSaved = s;
				}
				perfCollapsed = nc;
			}
			string fpsStr = rightPad("FPS:", 30) + strFps;
			ImGui::Text(fpsStr.c_str());

			static float history = 2.0f;
			static ScrollingBuffer sFrames;
			static ScrollingBuffer s60fps;
			static ScrollingBuffer s120fps;
			float t = static_cast<float>(now());

			sFrames.AddPoint(t, 1000.0f * static_cast<float>(timeSinceLastFrame));

			// sFrames.AddPoint(t, 1000.0f * timeSinceLastFrame);
			s60fps.AddPoint(t, 1000.0f / 60.0f);
			s120fps.AddPoint(t, 1000.0f / 120.0f);
			ImPlot::SetNextPlotLimitsX(t - history, t, ImGuiCond_Always);
			ImPlot::SetNextPlotLimitsY(0, 30, ImGuiCond_Always);

			if (ImPlot::BeginPlot("Timings", nullptr, nullptr, ImVec2(-1, 200))){

				auto x = &sFrames.Data[0].x;
				auto y = &sFrames.Data[0].y;
				ImPlot::PlotShaded("frame time(ms)", x, y, sFrames.Data.size(), -Infinity, sFrames.Offset, 2 * sizeof(float));

				ImPlot::PlotLine("16.6ms (60 FPS)", &s60fps.Data[0].x, &s60fps.Data[0].y, s60fps.Data.size(), s60fps.Offset, 2 * sizeof(float));
				ImPlot::PlotLine(" 8.3ms (120 FPS)", &s120fps.Data[0].x, &s120fps.Data[0].y, s120fps.Data.size(), s120fps.Offset, 2 * sizeof(float));

				ImPlot::EndPlot();
			}

			ImGui::End();
		}

		// Present: scale the rendered frame to the window, then draw the UI on
		// top at window size. The UI used to be drawn into the frame itself,
		// which only works while frame and window are the same size; this
		// also leaves the frame free of UI for screenshots.
		// A frame of another shape than the window (--render) is letterboxed,
		// not stretched.
		auto source = view.framebuffer;
		presentX = 0;
		presentY = 0;
		presentWidth  = this->windowWidth;
		presentHeight = this->windowHeight;
		if(source->width > 0 && source->height > 0 && this->windowWidth > 0 && this->windowHeight > 0){
			int64_t frameByWindow = int64_t(source->width) * this->windowHeight;
			int64_t windowByFrame = int64_t(this->windowWidth) * source->height;
			if(frameByWindow > windowByFrame){        // frame wider: bars top and bottom
				presentHeight = int(windowByFrame / source->width);
			}else if(frameByWindow < windowByFrame){  // frame taller: bars left and right
				presentWidth = int(frameByWindow / source->height);
			}
			presentX = (this->windowWidth  - presentWidth)  / 2;
			presentY = (this->windowHeight - presentHeight) / 2;
		}

		glBindFramebuffer(GL_FRAMEBUFFER, 0);
		glViewport(0, 0, this->windowWidth, this->windowHeight);
		if(presentWidth != this->windowWidth || presentHeight != this->windowHeight){
			glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
			glClear(GL_COLOR_BUFFER_BIT);
		}
		glBlitNamedFramebuffer(
			source->handle, 0,
			0, 0, source->width, source->height,
			presentX, presentY, presentX + presentWidth, presentY + presentHeight,
			GL_COLOR_BUFFER_BIT, GL_LINEAR);

		ImGui::Render();
		ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());


		glfwSwapBuffers(window);
		glfwPollEvents();

		fpsCounter++;
		frameCount++;
	}

	glfwDestroyWindow(window);
	glfwTerminate();

	// quick_exit, not exit: exit destroys the globals, such as the mutexes
	// and queues the loader and uploader threads are still using. The run
	// log is flushed after every line.
	fflush(stdout);
	std::quick_exit(exitCode);
}

shared_ptr<Framebuffer> Framebuffer::create(GLRenderer* renderer) {

	auto fbo = make_shared<Framebuffer>();
	fbo->renderer = renderer;

	glCreateFramebuffers(1, &fbo->handle);

	{ // COLOR ATTACHMENT 0

		auto texture = renderer->createTexture(fbo->width, fbo->height, GL_RGBA8);
		fbo->colorAttachments.push_back(texture);

		glNamedFramebufferTexture(fbo->handle, GL_COLOR_ATTACHMENT0, texture->handle, 0);
	}

	fbo->setSize(128, 128);

	return fbo;
}

shared_ptr<Texture> Texture::create(int width, int height, GLuint colorType, GLRenderer* renderer){

	GLuint handle;
	glCreateTextures(GL_TEXTURE_2D, 1, &handle);

	auto texture = make_shared<Texture>();
	texture->renderer = renderer;
	texture->handle = handle;
	texture->colorType = colorType;

	texture->setSize(width, height);

	return texture;
}

void Texture::setSize(int width, int height) {

	bool needsResize = this->width != width || this->height != height;

	if (needsResize) {

		glDeleteTextures(1, &this->handle);
		glCreateTextures(GL_TEXTURE_2D, 1, &this->handle);

		glTextureParameteri(this->handle, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
		glTextureParameteri(this->handle, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
		glTextureParameteri(this->handle, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
		glTextureParameteri(this->handle, GL_TEXTURE_MAG_FILTER, GL_NEAREST);

		glTextureStorage2D(this->handle, 1, this->colorType, width, height);

		this->width = width;
		this->height = height;
	}

}