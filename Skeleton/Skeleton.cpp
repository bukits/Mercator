//=============================================================================================
// Mintaprogram: Zöld háromszög. Ervenyes 2019. osztol.
//
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat, BOM kihuzando.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiveve
// - Mashonnan atvett programresszleteket forrasmegjeloles nelkul felhasznalni es
// - felesleges programsorokat a beadott programban hagyni!!!!!!! 
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak 
// A keretben nem szereplo GLUT fuggvenyek tiltottak.
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    : 
// Neptun : 
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================
#include "framework.h"

using namespace std;

const char * const vertexSource = R"(
	#version 330
	precision highp float;

	uniform int mercator = 1;
	uniform mat4 MVP_Mercator;
	uniform mat4 MVP_Globe;
	layout(location = 0) in vec2 vp;

	void main() {
		if (mercator == 1) {
			gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP_Mercator;
		} else if (mercator == 0) {
			gl_Position = vec4(vp.x, vp.y, 0, 1) * MVP_Globe;
		}
	}
)";

const char * const fragmentSource = R"(
	#version 330
	precision highp float;
	
	uniform vec3 color;
	out vec4 fragmentColor;

	void main() {
		fragmentColor = vec4(color, 1);
	}
)";

const float mu_0 = -20.0f;
const float mu_1 = 160.0f;
const float fi_0 = -85.0f;
const float fi_1 = 85.0f;
const float circleDiff = 20.0f;

vec2 ConvertDegree2Radian(vec2 degree) {
	return vec2(degree * (float)M_PI / 180.0f);
}

class Mercator {
	vec2 wCenter;
	vec2 wSize;
	vec2 xStart = vec2(fi_0, mu_0), xEnd = vec2(fi_0, mu_1), yStart = vec2(fi_0, mu_0), yEnd = vec2(fi_1, mu_0);
public:
	Mercator() {
		vec2 xDir = CalculateMercatorCoord(ConvertDegree2Radian(xEnd)) - CalculateMercatorCoord(ConvertDegree2Radian(xStart));
		vec2 yDir = CalculateMercatorCoord(ConvertDegree2Radian(yEnd)) - CalculateMercatorCoord(ConvertDegree2Radian(yStart));
		vec2 origo = CalculateMercatorCoord(ConvertDegree2Radian(vec2((fi_0 + fi_1) / 2.0f, (mu_0 + mu_1) / 2.0f)));
		float lengthX = length(xDir);
		float lengthY = length(yDir);
		wCenter = origo;
		wSize = vec2(lengthX, lengthY);
	}

	vec2 CalculateMercatorCoord(vec2 radian) {
		return { vec2(radian.y, logf((tanf(((float)M_PI / 4.0f) + (radian.x / 2.0f))))) };
	}

	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

	mat4 Vinv() { return TranslateMatrix(wCenter); }
	mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }
};

class Globe {
	vec2 wCenter;
	vec2 wSize;
public:
	Globe() : wCenter(0.8,1.0), wSize(5, 5) {}
	vec3 CalculatePolarCoord(vec2 radian) {
		float x = cos(radian.y) * cos(radian.x);
		float y = sin(radian.y) * cos(radian.x);
		float z = sin(radian.x);
		return vec3(x, y, z);
	}

	vec2 Proj2Plane(vec3 polar) {
		return vec2(polar.x, polar.y);
	}

	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }
};

GPUProgram gpuProgram;
Mercator mercator;
Globe globe;
const int nTesselatedVertices = 100;
bool modelMercator = true;

class Geometry {
protected:
	unsigned int vao, vbo;
	vec3 color;
public:
	virtual void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() {
		mat4 VPTransformMercator = mercator.V() * mercator.P();
		gpuProgram.setUniform(VPTransformMercator, "MVP_Mercator");

		mat4 VPTransformGlobe = globe.V() * globe.P();
		gpuProgram.setUniform(VPTransformGlobe, "MVP_Globe");
	}
};

class Earth : public Geometry {	
	vector<vec2> cps;
public:
	Earth(vector<vec2> rectCoord, vec3 color) {
		this->color = color;
	}

	void Create() {
		Geometry::Create();
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2, NULL);
	}

	void Draw() {
		vector<float> vertexData;
		
		
	}
};


class Circle : public Geometry {
	vec2 startCp, endCp;
public:
	Circle(vec2 start, vec2 end, vec3 color) {
		this->color = color;
		
		vec2 startRadian = ConvertDegree2Radian(start);
		vec2 endRadian = ConvertDegree2Radian(end);

		this->startCp = mercator.CalculateMercatorCoord(startRadian);
		this->endCp = mercator.CalculateMercatorCoord(endRadian);
	}

	void Create() {
		Geometry::Create();
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL);
	}

	void Draw() {
		Geometry::Draw();

		vector<float> vertexData;
		for (int i = 0; i < nTesselatedVertices; i++) {
			float tNormalized = (float)i / (nTesselatedVertices - 1);
			vec2 p = startCp + (endCp - startCp) * tNormalized;
			vertexData.push_back(p.x);
			vertexData.push_back(p.y);
		}
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_STATIC_DRAW);
		gpuProgram.setUniform(this->color, "color");
		glLineWidth(1.0f);
		glDrawArrays(GL_LINE_STRIP, 0, nTesselatedVertices);
	}
};

class Continent : public Geometry {
	vector<vec4> splineCps;
	vector<float> ts;

	unsigned int vaoCtrlPoints, vboCtrlPoints;

	vec4 s(int i, float t) {
		vec4 a_i, b_i, c_i;
		if (i == 0) {
			a_i = (splineCps[i + 1] - splineCps[i]) * (1 / ((ts[i + 1] - ts[i]) * (ts[i + 1] - tEnd()))) -
				(splineCps.back() - splineCps[i]) * (1 / ((tEnd() - ts[i]) * (ts[i + 1] - tEnd())));
			b_i = (splineCps.back() - splineCps[i]) * (1 / (tEnd() - ts[i])) -
				(splineCps[i + 1] - splineCps[i]) * (1 / ((ts[i + 1] - ts[i]) * (ts[i + 1] - tEnd()))) * (tEnd() - ts[i]) +
				(splineCps.back() - splineCps[i]) * (1 / (ts[i + 1] - tEnd()));
		}
		else if (i == splineCps.size() - 1) {
			a_i = (splineCps.front() - splineCps[i]) * (1 / ((tStart() - ts[i]) * (tStart() - ts[i - 1]))) -
				(splineCps[i - 1] - splineCps[i]) * (1 / ((ts[i - 1] - ts[i]) * (tStart() - ts[i - 1])));
			b_i = (splineCps[i - 1] - splineCps[i]) * (1 / (ts[i - 1] - ts[i])) -
				(splineCps.front() - splineCps[i]) * (1 / ((tStart() - ts[i]) * (tStart() - ts[i - 1]))) * (ts[i - 1] - ts[i]) +
				(splineCps[i - 1] - splineCps[i]) * (1 / (tStart() - ts[i - 1]));
		}
		else {
			a_i = (splineCps[i + 1] - splineCps[i]) * (1 / ((ts[i + 1] - ts[i]) * (ts[i + 1] - ts[i - 1]))) -
				(splineCps[i - 1] - splineCps[i]) * (1 / ((ts[i - 1] - ts[i]) * (ts[i + 1] - ts[i - 1])));
			b_i = (splineCps[i - 1] - splineCps[i]) * (1 / (ts[i - 1] - ts[i])) -
				(splineCps[i + 1] - splineCps[i]) * (1 / ((ts[i + 1] - ts[i]) * (ts[i + 1] - ts[i - 1]))) * (ts[i- 1] - ts[i])  +
				(splineCps[i - 1] - splineCps[i]) * (1 / (ts[i + 1] - ts[i - 1]));
		}
		c_i = splineCps[i];
		return a_i * powf(t - ts[i], 2) + b_i * (t - ts[i]) + c_i;
	}

	void AddControlPoint(vec2 cp) {
		ts.push_back((float)splineCps.size());
		splineCps.push_back(vec4(cp.x, cp.y, 0, 1));
	}

	float tStart() { return ts[0]; }
	float tEnd() { return ts[splineCps.size() - 1]; }
public:
	Continent(vector<vec2> inputCoords, vec3 color) {
		for (auto coord : inputCoords) {
			vec2 worldRadian = ConvertDegree2Radian(coord);
			vec2 worldCoord;
			if (modelMercator) {
				worldCoord = mercator.CalculateMercatorCoord(worldRadian);
			}
			else {
				vec3 polar = globe.CalculatePolarCoord(worldRadian);
				worldCoord = globe.Proj2Plane(polar);
			}
			AddControlPoint(worldCoord);
		}
		this->color = color;
	}

	vec4 r(float t) {
		vec4 ret(0, 0, 0, 0);
		for (int index = 0; index < splineCps.size() - 1; ++index) {
			if (ts[index] <= t && t <= ts[index + 1]) {
				return ret = (s(index, t) * (ts[index + 1] - t) + s(index + 1, t) * (t - ts[index])) / (ts[index + 1] - ts[index]);
			}
		}
		if (t == tEnd()) {
			int index = splineCps.size() - 1;
			return ret = (s(index, t) * (tStart() - t) + s(0, t) * (t - ts[index])) / (tStart() - ts[index]);
		}
		return ret;
	}

	void Create() {
		Geometry::Create();
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 2 * sizeof(float), NULL);

		glGenVertexArrays(1, &vaoCtrlPoints);
		glBindVertexArray(vaoCtrlPoints);

		glGenBuffers(1, &vboCtrlPoints);
		glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), NULL);
	}

	void Draw() {
		Geometry::Draw();
		glBindVertexArray(vaoCtrlPoints);
		glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
		glBufferData(GL_ARRAY_BUFFER, splineCps.size() * 4 * sizeof(float), &splineCps[0], GL_DYNAMIC_DRAW);
		gpuProgram.setUniform(color, "color");
		glPointSize(10.0f);
		glDrawArrays(GL_POINTS, 0, splineCps.size());

		vector<float> vertexData;
		for (int i = 0; i < nTesselatedVertices; i++) {
			float tNormalized = (float)i / (nTesselatedVertices - 1);
			float t = tStart() + (tEnd() - tStart()) * tNormalized;
			vec4 wVertex = r(t);
			vertexData.push_back(wVertex.x);
			vertexData.push_back(wVertex.y);
		}
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_DYNAMIC_DRAW);
		gpuProgram.setUniform(color, "color");
		glLineWidth(2.0f);
		glDrawArrays(GL_LINE_STRIP, 0, nTesselatedVertices);
	}
};

vector<Geometry*> entities;

vector<vec2> eurasiaCoords{ vec2(36.0f, 0.0f), vec2(42.0f, 0.0f), vec2(47.0f, -3.0f), vec2(61.0f, 6.0f), vec2(70.0f, 28.0f),
								vec2(65.0f, 44.0f), vec2(76.0f, 113.0f), vec2(60.0f, 160.0f), vec2(7.0f, 105.0f), vec2(19.0f, 90.0f),
								vec2(4.0f, 80.0f), vec2(42.0f, 13.0f) };
vec3 eurasiaColor(0.05f, 1.1f, 0.0f);

vector<vec2> africaCoords{ vec2(33.0f, -5.0f), vec2(17.0f, -16.0f), vec2(3.0f, 6.0f),
						   vec2(-35.0f, 19.0f), vec2(-3.0f, 40.0f), vec2(10.0f, 53.0f), vec2(30.0f, 33.0f) };
vec3 africaColor(1.8f, 0.8f, 0.0f);

void CreateEntities() {
	
	/*vector<vec2> earthCoords = { vec2(fi_0, mu_0), vec2(fi_0, mu_1), vec2(fi_1, mu_1), vec2(fi_1, mu_0) };
	vec3 earthColor(0.05f, 0.0f, 0.9f);*/

	/*Geometry* earth = new Earth(earthCoords, earthColor);
	entities.push_back(earth);*/

	Geometry* eurasia = new Continent(eurasiaCoords, eurasiaColor);
	entities.push_back(eurasia);

	Geometry* africa = new Continent(africaCoords, africaColor);
	entities.push_back(africa);

	/*vec3 circleColor(1.0f, 1.0f, 1.0f);
	for (float latitude = -90.0f + circleDiff; latitude < 90.0f; latitude += circleDiff) {
		entities.push_back(new Circle(vec2(latitude, mu_0), vec2(latitude, mu_1), circleColor));
	}

	for (float longitude = mu_0 + circleDiff; longitude < mu_1; longitude += circleDiff) {
		entities.push_back(new Circle(vec2(fi_0, longitude), vec2(fi_1, longitude), circleColor));
	}*/

	for (auto entity : entities) entity->Create();
}

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2.0f);

	CreateEntities();

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void CreateMercatorWorld() {
	entities.clear();
	CreateEntities();
	gpuProgram.setUniform(0, "mercator");
	glutPostRedisplay();
	modelMercator = false;
}

void CreateGlobeWorld() {
	entities.clear();
	CreateEntities();
	gpuProgram.setUniform(1, "mercator");
	glutPostRedisplay();
	modelMercator = true;
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);
	for (auto entity : entities) entity->Draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'm') {
		if (modelMercator) CreateGlobeWorld();
		else CreateMercatorWorld();
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouseMotion(int pX, int pY) {}

void onMouse(int button, int state, int pX, int pY) {}

void onIdle() {}
