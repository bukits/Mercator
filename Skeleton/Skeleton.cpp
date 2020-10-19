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

	#define PI 3.1415926538

	precision highp float;

	uniform int isMercator = 1;

	uniform mat4 MVP;
	layout(location = 0) in vec2 vp;

	float degreeToRadian(float degree) {
		return degree * PI / 180;
	}

	vec2 ConvertDegree2Radian(vec2 degree) {
		return degree * PI / 180;
	}

	vec2 toMercator(vec2 coordDegree) {
		vec2 coordRadian = ConvertDegree2Radian(coordDegree);
		return vec2(coordRadian.y, log(tan((PI / 4) + (coordRadian.x / 2))));
	}

	vec2 toGlobe(vec2 coordDegree) {
		vec2 coordRadian = ConvertDegree2Radian(coordDegree);
		coordRadian.y += degreeToRadian(20);
		float x = cos(coordRadian.y) * cos(coordRadian.x);
		float y = sin(coordRadian.y) * cos(coordRadian.x);
		float z = sin(coordRadian.x);
		return vec2(-x, z);
	}

	void main() {
		vec2 outCoord;
		if (isMercator == 1) {
			outCoord = toMercator(vp);
		} else if (isMercator == 0) {
			outCoord = toGlobe(vp);
		}
		gl_Position = vec4(outCoord.x, outCoord.y, 0, 1) * MVP;
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

const float lambda_0 = -20.0f;
const float lambda_1 = 160.0f;
const float fi_0 = -85.0f;
const float fi_1 = 85.0f;
const float circleDiff = 20.0f;

const int earthRadius = 6371;

vec2 ConvertDegree2Radian(vec2 degree) {
	return degree * (float)M_PI / 180.0f;
}

vec2 ConvertRadian2Degree(vec2 radian) {
	return radian * 180.0f / (float)M_PI;
}

float DegreeToRadian(float degree) {
	return degree * (float)M_PI / 180.0f;
}

class Mercator {
	vec2 wSize;
	vec2 xStart = vec2(fi_0, lambda_0), xEnd = vec2(fi_0, lambda_1), yStart = vec2(fi_0, lambda_0), yEnd = vec2(fi_1, lambda_0);
public:
	vec2 wCenter;

	Mercator() {
		vec2 xDir = CalculateMercator(ConvertDegree2Radian(xEnd)) - CalculateMercator(ConvertDegree2Radian(xStart));
		vec2 yDir = CalculateMercator(ConvertDegree2Radian(yEnd)) - CalculateMercator(ConvertDegree2Radian(yStart));
		vec2 origo = CalculateMercator(ConvertDegree2Radian(vec2((fi_0 + fi_1) / 2.0f, (lambda_0 + lambda_1) / 2.0f)));
		float lengthX = length(xDir);
		float lengthY = length(yDir);
		wCenter = origo;
		wSize = vec2(lengthX, lengthY);
	}

	vec2 CalculateMercator(vec2 radian) {
		return vec2(radian.y, logf((tanf(((float)M_PI / 4.0f) + (radian.x / 2.0f)))));
	}

	vec2 CalculateMercatorInverse(vec2 mercator) {
		return vec2(2 * atanf(expf(mercator.y)) - (float)M_PI / 2.0f, mercator.x);
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
	Globe() : wSize(2, 2), wCenter(0, 0) {}

	vec2 CalculatePolarInverse(vec3 polar) {
		float fi = asinf(polar.z);
		float lambda = acosf(polar.x / cosf(fi));
		lambda -= DegreeToRadian(20.0f);
		return vec2(fi, lambda);
	}

	vec3 CalculatePolar(vec2 coordDegree) {
		float x = cos(coordDegree.y) * cos(coordDegree.x);
		float y = sin(coordDegree.y) * cos(coordDegree.x);
		float z = sin(coordDegree.x);
		return vec3(x, y, z);
	}

	vec3 OrthogonalProjInverse(vec2 ndc) {
		float x = ndc.x;
		float y = ndc.y;
		float z = sqrtf(1.0f - powf(x, 2) - powf(y, 2));
		return vec3(-x, z, y);
	}

	mat4 V() { return TranslateMatrix(-wCenter); }
	mat4 P() { return ScaleMatrix(vec2(2 / wSize.x, 2 / wSize.y)); }

	mat4 Vinv() { return TranslateMatrix(wCenter); }
	mat4 Pinv() { return ScaleMatrix(vec2(wSize.x / 2, wSize.y / 2)); }
};

GPUProgram gpuProgram;
Mercator mercator;
Globe globe;
const int nTesselatedVertices = 200;
bool modelMercator = true;

class Geometry {
protected:
	unsigned int vao, vbo;
	vec3 mainColor;
public:
	virtual void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
	}
	virtual void Draw() {
		mat4 VPTransform;
		if (modelMercator) {
			VPTransform = mercator.V() * mercator.P();
		}
		else {
			VPTransform = globe.V() * globe.P();
		}
		gpuProgram.setUniform(VPTransform, "MVP");
	}
};

class Earth : public Geometry {	
	vector<vec2> cps;
public:
	Earth(const vector<vec2>& rectCoord, vec3 color) {
		this->mainColor = color;
		for (auto degree : rectCoord) {
			cps.push_back(degree);
		}
	}

	void Create() {
		Geometry::Create();
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 0, NULL);
	}

	void Draw() {
		Geometry::Draw();
		vector<float> vertexData;
		vertexData.push_back((fi_0 + fi_1) / 2.0f);
		vertexData.push_back((lambda_0 + lambda_1) / 2.0f);
		for (int j = 0; j < cps.size(); ++j) {
			for (int i = 0; i < nTesselatedVertices; i++) {
				float tNormalized = (float)i / (nTesselatedVertices - 1);
				vec2 p;
				if (j == 3) p = cps[j] + (cps[0] - cps[j]) * tNormalized;
				else p = cps[j] + (cps[j + 1] - cps[j]) * tNormalized;
				vertexData.push_back(p.x);
				vertexData.push_back(p.y);
			}
		}
		
		glBindVertexArray(vao);
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, vertexData.size() * sizeof(float), &vertexData[0], GL_STATIC_DRAW);
		gpuProgram.setUniform(this->mainColor, "color");
		glDrawArrays(GL_TRIANGLE_FAN, 0, ((nTesselatedVertices * 2 + 2) * cps.size()) / 2);
	}
};

class Curve : public Geometry {
	unsigned int vaoCtrlPoints = 0, vboCtrlPoints = 0;
protected:
	vector<vec4> splineCps;
	vec3 pointColor;
public:
	Curve(vec3 lineColor, vec3 pointColor) {
		this->mainColor = lineColor;
		this->pointColor = pointColor;
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

	virtual vec4 r(float t) { return splineCps[0]; }
	virtual float tStart() { return 0; }
	virtual float tEnd() { return 1; }
	virtual void AddControlPoint(vec2 cp) = 0;

	void Draw() {
		Geometry::Draw();
		if (splineCps.size() > 0) {
			glBindVertexArray(vaoCtrlPoints);
			glBindBuffer(GL_ARRAY_BUFFER, vboCtrlPoints);
			glBufferData(GL_ARRAY_BUFFER, splineCps.size() * 4 * sizeof(float), &splineCps[0], GL_DYNAMIC_DRAW);
			gpuProgram.setUniform(this->pointColor, "color");
			glPointSize(10.0f);
			glDrawArrays(GL_POINTS, 0, splineCps.size());
		}

		if (splineCps.size() >= 2) {
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
			gpuProgram.setUniform(this->mainColor, "color");
			glLineWidth(2.0f);
			glDrawArrays(GL_LINE_STRIP, 0, nTesselatedVertices);
		}
	}
};

class Path : public Curve {
	vector<float> ts;

	//forrás: https://keisan.casio.com/exec/system/1224587128
	float CalculateDistance(vec2 p_1, vec2 p_2) {
		float delta = abs(p_2.x - p_1.x);
		return earthRadius * acosf((sinf(p_1.y) * sinf(p_2.y)) + (cosf(p_1.y) * cosf(p_2.y) * cosf(delta)));
	}

	void ToConsole(vec2 latitudeLongitude) {
		printf("Longitude: %f, Latitude: %f\n", latitudeLongitude.y, latitudeLongitude.x);
		if (splineCps.size() > 0) {
			vec4 lastCoord = splineCps.back();
			float distance = CalculateDistance(latitudeLongitude, vec2(lastCoord.x, lastCoord.y));
			printf("Distance: %f km\n", distance);
		}
	}

	vec4 Slerp(float t, float t_0, vec4 pStart, vec4 pEnd) {
		t = t - t_0;
		float d = acosf(dot(normalize(vec2(pStart.x, pStart.y)), normalize(vec2(pEnd.x, pEnd.y))));
		vec3 sphereStart = globe.CalculatePolar(ConvertDegree2Radian(vec2(pStart.x, pStart.y)));
		vec3 sphereEnd = globe.CalculatePolar(ConvertDegree2Radian(vec2(pEnd.x, pEnd.y)));
		vec4 r = (pStart * sinf((1 - t) * d) / sinf(d)) + (pEnd * sinf(t * d) / sinf(d));
		return vec4(r.x, r.y, r.z);
	}
public:
	Path(vec3 lineColor, vec3 pointColor) : Curve(lineColor, pointColor) {}

	void AddControlPoint(vec2 cp) {
		vec2 latitudeLongitude;
		if (modelMercator) {
			vec4 wVertex = vec4(cp.x, cp.y, 0, 1) * mercator.Pinv() * mercator.Vinv();
			latitudeLongitude = mercator.CalculateMercatorInverse(vec2(wVertex.x, wVertex.y));
		}
		else {
			vec4 wVertex = vec4(cp.x, cp.y, 0, 1) * globe.Pinv() * globe.Vinv();
			latitudeLongitude = globe.CalculatePolarInverse(globe.OrthogonalProjInverse(vec2(wVertex.x, wVertex.y)));
		}
		vec2 degreeLatLong = ConvertRadian2Degree(latitudeLongitude);
		ToConsole(degreeLatLong);
		ts.push_back((float)splineCps.size());
		splineCps.push_back(vec4(degreeLatLong.x, degreeLatLong.y));
	}

	 float tStart() { return ts[0]; }
	 float tEnd() { return ts[splineCps.size() - 1]; }

	 vec4 r(float t) {
		 vec4 ret(0, 0, 0, 0);
		 for (int i = 0; i < splineCps.size() - 1; i++) {
			 if (ts[i] <= t && t <= ts[i + 1]) {
				 return ret = Slerp(t, ts[i], splineCps[i], splineCps[i + 1]);
			 }
		 }
		 return ret;
	 }
};

class Circle : public Geometry {
	vec2 startCp, endCp;
public:
	Circle(vec2 start, vec2 end, vec3 color) {
		this->mainColor = color;
		
		this->startCp = start;
		this->endCp = end;
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
		gpuProgram.setUniform(this->mainColor, "color");
		glLineWidth(1.0f);
		glDrawArrays(GL_LINE_STRIP, 0, nTesselatedVertices);
	}
};

class Continent : public Curve {
	vector<float> ts;

	vec4 s(int i, float t) {
		vec4 a_i, b_i, c_i;
		if (i == 0) {
			a_i = (splineCps[i + 1] - splineCps[i]) * (1 / ((ts[i + 1] - ts[i]) * (ts[i + 1] - ts[splineCps.size() - 1]))) -
				(splineCps.back() - splineCps[i]) * (1 / ((ts[splineCps.size() - 1] - ts[i]) * (ts[i + 1] - ts[splineCps.size() - 1])));
			b_i = (splineCps.back() - splineCps[i]) * (1 / (ts[splineCps.size() - 1] - ts[i])) -
				(splineCps[i + 1] - splineCps[i]) * (1 / ((ts[i + 1] - ts[i]) * (ts[i + 1] - ts[splineCps.size() - 1]))) * (ts[splineCps.size() - 1] - ts[i]) +
				(splineCps.back() - splineCps[i]) * (1 / (ts[i + 1] - ts[splineCps.size() - 1]));
		}
		else if (i == splineCps.size() - 1) {
			a_i = (splineCps[1] - splineCps[i]) * (1 / ((ts[1] - ts[i]) * (ts[1] - ts[i - 1]))) -
				(splineCps[i - 1] - splineCps[i]) * (1 / ((ts[i - 1] - ts[i]) * (ts[1] - ts[i - 1])));
			b_i = (splineCps[i - 1] - splineCps[i]) * (1 / (ts[i - 1] - ts[i])) -
				(splineCps[1] - splineCps[i]) * (1 / ((ts[1] - ts[i]) * (ts[1] - ts[i - 1]))) * (ts[i - 1] - ts[i]) +
				(splineCps[i - 1] - splineCps[i]) * (1 / (ts[1] - ts[i - 1]));
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
	Continent(const vector<vec2>& inputCoords, vec3 color) : Curve(color, color) {
		for (auto coord : inputCoords) {
			AddControlPoint(coord);
		}
		AddControlPoint(inputCoords[0]);
	}

	vec4 r(float t) {
		vec4 ret(0, 0, 0, 0);
		for (int index = 0; index < splineCps.size() - 1; ++index) {
			if (ts[index] <= t && t <= ts[index + 1]) {
				return ret = (s(index, t) * (ts[index + 1] - t) + s(index + 1, t) * (t - ts[index])) / (ts[index + 1] - ts[index]);
			}
		}
		return ret;
	}
};

vector<Geometry*> entities;
Curve* path;

vector<vec2> eurasiaCoords{ vec2(36.0f, 0.0f), vec2(42.0f, 0.0f), vec2(47.0f, -3.0f), vec2(61.0f, 6.0f), vec2(70.0f, 28.0f),
								vec2(65.0f, 44.0f), vec2(76.0f, 113.0f), vec2(60.0f, 160.0f), vec2(7.0f, 105.0f), vec2(19.0f, 90.0f),
								vec2(4.0f, 80.0f), vec2(42.0f, 13.0f) };

vector<vec2> africaCoords{ vec2(33.0f, -5.0f), vec2(17.0f, -16.0), vec2(3.0f, 6.0f),
						   vec2(-35.0f, 19.0f), vec2(-3.0f, 40.0f), vec2(10.0f, 53.0f), vec2(30.0f, 33.0f) };

vector<vec2> earthCoords{ vec2(fi_0, lambda_0), vec2(fi_0, lambda_1), vec2(fi_1, lambda_1), vec2(fi_1, lambda_0) };
	
vec3 africaColor(1.8f, 0.8f, 0.0f);
vec3 circlecolor(1.0f, 1.0f, 1.0f);
vec3 eurasiaColor(0.05f, 1.1f, 0.0f);
vec3 earthColor(0.05f, 0.0f, 0.9f);
vec3 pathLineColor(1.0f, 1.0f, 0.0f);
vec3 pathPointColor(1.0f, 0.0f, 0.0f);

void CreateEntities() {
	
	Geometry* earth = new Earth(earthCoords, earthColor);
	entities.push_back(earth);

	path = new Path(pathLineColor, pathPointColor);
	entities.push_back(path);

	Geometry* eurasia = new Continent(eurasiaCoords, eurasiaColor);
	entities.push_back(eurasia);

	Geometry* africa = new Continent(africaCoords, africaColor);
	entities.push_back(africa);

	for (float latitude = -90.0f; latitude <= 90.0f; latitude += circleDiff) {
		entities.push_back(new Circle(vec2(latitude, lambda_0), vec2(latitude, lambda_1 + 20), circlecolor));
	}

	for (float longitude = lambda_0; longitude <= lambda_1 + 20; longitude += circleDiff) {
		entities.push_back(new Circle(vec2(fi_0, longitude), vec2(fi_1, longitude), circlecolor));
	}

	for (auto entity : entities) entity->Create();
}

void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);
	glLineWidth(2.0f);

	CreateEntities();

	gpuProgram.create(vertexSource, fragmentSource, "outColor");
}

void onDisplay() {
	glClearColor(0, 0, 0, 0);
	glClear(GL_COLOR_BUFFER_BIT);
	for (auto entity : entities) entity->Draw();
	glutSwapBuffers();
}

void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'm') {
		modelMercator = !modelMercator;
		gpuProgram.setUniform(modelMercator, "isMercator");
		glutPostRedisplay();
	}
}

void onKeyboardUp(unsigned char key, int pX, int pY) {}

void onMouseMotion(int pX, int pY) {}

void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {
		float cX = 2.0f * pX / windowWidth - 1;
		float cY = 1.0f - 2.0f * pY / windowHeight;
		path->AddControlPoint(vec2(cX, cY));
		glutPostRedisplay();
	}
}

void onIdle() {}
