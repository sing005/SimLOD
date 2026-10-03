
#pragma once

#include <string>
#include <unordered_map>
#include <map>

#include "glm/common.hpp"

using namespace std;

struct Runtime{

	struct GuiItem{
		uint32_t type = 0;
		float min = 0.0;
		float max = 1.0;
		float oldValue = 0.5;
		float value = 0.5;
		string label = "";
	};

	inline static vector<int> keyStates = vector<int>(65536, 0);
	inline static glm::dvec2 mousePosition = {0.0, 0.0};
	inline static int mouseButtons = 0;
	// torch mode: the mouse aims the torch, so mouse moves do not turn the camera
	inline static bool torchMode = false;
	inline static bool showGUI = true;
	inline static bool showPerfGraph = true;

	Runtime(){
		
	}

	static Runtime* getInstance(){
		static Runtime* instance = new Runtime();

		return instance;
	}

};