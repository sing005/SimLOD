// include/OrbitControls.h

#pragma once

#include <iostream>

#include "glm/common.hpp"
#include "glm/matrix.hpp"
#include <glm/gtx/transform.hpp>

#include "Runtime.h"

using namespace std;

struct OrbitControls{

	double yaw = 0.0;
	double pitch = 0.0;
	double radius = 2;
	glm::dvec3 target = {0.0, 0.0, 0.0};
	glm::dmat4 world;

	bool isLeftDown = false;
	bool isRightDown = false;

	glm::dvec2 mousePos;

	OrbitControls(){
	
	}

	glm::dvec3 getDirection(){
		auto rotation = getRotation();

		auto dir = rotation * glm::dvec4(0, 1, 0, 1.0);

		return dir;
	}

	glm::dvec3 getPosition(){
		auto dir = getDirection();

		auto pos = target - (radius * dir);

		//cout << 
		//	"[" << target.x << ", " << target.y << ", " << target.z << "] - " <<
		//	"[" << pos.x << ", " << pos.y << ", " << pos.z << "] - " <<
		//	"[" << dir.x << ", " << dir.y << ", " << dir.z << "]" << endl;

		return pos;
	}

	glm::dmat4 getRotation(){
		glm::dvec3 up    = {0, 0, 1};
		glm::dvec3 right = {1, 0, 0};

		auto rotYaw = glm::rotate(yaw, up);
		auto rotPitch = glm::rotate(pitch, right);

		//auto rotation = rotYaw;
		//auto rotation = rotPitch * rotYaw;
		// apply_markers2.ps1: update() builds world as rotYaw * rotPitch; this
		// had them the other way round, so getPosition() (Copy camera) did not
		// report the camera the renderer was actually using.
		auto rotation = rotYaw * rotPitch;

		return rotation;
	}

	void translate_local(double x, double y, double z){
		auto _pos = glm::dvec3(0, 0, 0);
		auto _right = glm::dvec3(1, 0, 0);
		auto _forward = glm::dvec3(0, 1, 0);
		auto _up = glm::dvec3(0, 0, 1);

		_pos = world * glm::dvec4(_pos, 1);
		_right = world * glm::dvec4(_right, 1);
		_forward = world * glm::dvec4(_forward, 1);
		_up = world * glm::dvec4(_up, 1);

		_right = glm::normalize(_right - _pos) * x;
		_forward = glm::normalize(_forward - _pos) * y;
		_up = glm::normalize(_up - _pos) * (-z);

		this->target = this->target + + _right + _forward + _up;
	}

	void onMouseButton(int button, int action, int mods){
		//cout << "button: " << button << ", action: " << action << ", mods: " << mods << endl;

		if(button == 0 && action == 1){
			isLeftDown = true;
		}else if(action == 0){
			isLeftDown = false;
		}

		if(button == 1 && action == 1){
			isRightDown = true;
		}else if(action == 0){
			isRightDown = false;
		}
	}

	void onMouseMove(double xpos, double ypos){

		bool selectActive = Runtime::keyStates[342] > 0;
		if(selectActive){
			return;
		}

		
		glm::dvec2 newMousePos = {xpos, ypos};
		glm::dvec2 diff = newMousePos - mousePos;

		if(isLeftDown){
			yaw -= double(diff.x) / 400.0;
			pitch -= double(diff.y) / 400.0;
		}else if(isRightDown){
			auto ux = diff.x / 1000.0;
			auto uy = diff.y / 1000.0;

			// pan scales with the orbit radius; in walk mode the radius is
			// pinned near zero, so use the walk step instead (10 steps per
			// 1000 px) or right-drag would not move the camera at all
			double panScale = walkMode ? walkStep * 10.0 : radius;
			translate_local(-ux * panScale, uy * panScale, 0);
		}
		
		mousePos = newMousePos;
	}

	void onMouseScroll(double xoffset, double yoffset){
		// horizontal-only scroll (trackpad swipe, tilt wheel) used to count as
		// "backwards" in walk mode and "zoom in" when orbiting
		if(yoffset == 0.0) return;

		// +1 in, -1 out. Orbiting from a distance, the wheel zooms by scaling
		// the radius. Standing inside the forest - radius under half a metre,
		// which is what Go to centre sets - scaling 0.001 m by 1.1 moves the
		// camera a tenth of a millimetre, so the wheel WALKS instead: a step
		// along the view direction. (apply_navigate.ps1)
		if(walkMode){
			// translate_local moves the TARGET; the camera sits one radius
			// behind it. Pinning the radius each step is what keeps the two
			// together - without it, looking around lets the camera drift
			// back, radius grows past the old 0.5 m test, and the wheel
			// silently reverts to zooming, which at a steep pitch reads as
			// rising rather than advancing. (fix_walk_targets.ps1)
			double step = walkStep * (yoffset > 0.0 ? 1.0 : -1.0);
			translate_local(0.0, 0.0, step);   // fix_walk_axis: z is forward, y is screen-up
			radius = 0.001;
		}else if(yoffset < 0.0){
			radius = radius * 1.1;
		}else{
			radius = radius / 1.1;
		}
	}
	bool   walkMode = false;   // set by Go to centre / the Walk button
	double walkStep = 0.75;    // metres per wheel notch when walking

	void update(){
		glm::dvec3 up    = {0, 0, 1};
		glm::dvec3 right = {1, 0, 0};

		auto translateRadius = glm::translate(
			glm::dmat4(), 
			glm::dvec3(0.0, 0.0, radius));
		auto translateTarget = glm::translate(glm::dmat4(), target);
		auto rotYaw = glm::rotate(yaw, up);
		auto rotPitch = glm::rotate(pitch, right);

		auto flip = glm::dmat4(
			1.0, 0.0, 0.0, 0.0,
			0.0, 0.0, 1.0, 0.0,
			0.0, -1.0, 0.0, 0.0,
			0.0, 0.0, 0.0, 1.0
		);

		world = translateTarget * rotYaw * rotPitch * flip * translateRadius;
	}

};