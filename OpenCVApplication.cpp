// OpenCVApplication.cpp : Defines the entry point for the console application.

#include "stdafx.h"
#include "common.h"
#include <iostream>
#include <stdio.h>
#include <random>
#include <vector>
#include <algorithm>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/imgcodecs.hpp"

#define BLACK 0
#define WHITE 255
#define MINIMUM_DEPTH_NOTE 26
#define WHITE_GAP 5
#define LINE_WIDTH 2
#define LINE_SPACING 19


#define	C8 108
#define	B7 107
#define	Bb7 106
#define	A7 105
#define	Ab7 104
#define	G7 103
#define	Gb7 102
#define	F7 101
#define	E7 100
#define	Eb7 99
#define	D7 98
#define	Db7 97
#define	C7 96,
#define	Bb6 94,
#define	A6 93,
#define	Ab6 92,
#define	G6 91
#define	Gb6 90
#define	F6 89
#define	E6 88
#define	D6 86
#define	C6 84
#define	B5 83
#define	A5 81
#define	G5 79
#define	F5 77
#define	E5 76
#define	D5 74
#define	C5 72
#define	B4 71
#define	A4 69
#define	G4 67
#define	F4 65
#define	E4 64
#define	D4 62
#define	Db4 61
#define	B3 59
#define	Bb3 58
#define	A3 57
#define	Ab3 56
#define	G3 55
#define	Gb 54
#define	F3 53
#define	E3 52
#define	Eb3 51
#define	D3 50
#define	Db3 49
#define	C3 48
#define	B2 47
#define	Bb2 46
#define	A2 45
#define	Ab2 44
#define	G2 43
#define	Gb2 42
#define	F2 41
#define	E2 40
#define	Eb2 39
#define	D2 38
#define	Db2 37
#define	C2 36,
#define	B1 35
#define	Bb1 34
#define	A1 33
#define	Ab1 32
#define	G1 31
#define	Gb1 30
#define	F1 29
#define	E1 28
#define	Eb1 27
#define	D1 26
#define	Db1 25
#define	C1 24
#define	B0 23
#define	Bb0 22
#define	A0 21

using namespace std;
using namespace cv;

typedef struct {
	int yOffset;
	int inaltime;
} dimensiuni_portativ;

typedef struct {
	Point p;
	char* noteName;
}MatchedNote;

typedef struct {
	MatchedNote note;
	double score;
}MatchedNoteWithScore;

typedef struct {
	int start;
	int end;
}Subset;

typedef struct {
	Mat templ;
	char* type;
}TemplateType;

//pt matching
Mat templ; Mat result;

deque<Mat> portativeImagine;
deque<Mat> portativeEmpty;
deque<Mat> portativeFitrate;

vector<Subset> areas[3];
vector<MatchedNote> notes[3];
deque<TemplateType> templates;

int nrPortative;

bool isInside(Mat img, int i, int j) {
	if (i >= 0 && i < img.rows && j >= 0 && j < img.cols)
		return true;
	else
		return false;
}

int* histogram(Mat src) {
	int* hist_vect = (int*)calloc(256, sizeof(int));
	int nr_acc, dim_out;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			for (int k = 0; k < 256; k++) {
				if (src.at<uchar>(i, j) == k)
					hist_vect[k]++;
			}
		}
	}
	return hist_vect;
}

void DrawCross(Mat& img, Point p, int size, Scalar color, int thickness)
{
	line(img, Point(p.x - size / 2, p.y), Point(p.x + size / 2, p.y), color, thickness, 8);
	line(img, Point(p.x, p.y - size / 2), Point(p.x, p.y + size / 2), color, thickness, 8);
}

void showHorizontalHistogram(const std::string& name, int* hist, const int  hist_rows, const int hist_cols) {
	Mat imgHist(hist_rows, hist_cols, CV_8UC1, CV_RGB(255, 255, 255)); // constructs a white image	
	for (int i = 0; i < hist_rows; i++) {
		for (int j = 0; j < hist[i]; j++) {
			imgHist.at<uchar>(i, j) = BLACK;
		}
	}
	imshow(name, imgHist);
}

void showVerticalHistogram(const std::string& name, int* hist, const int  hist_rows, const int hist_cols) {
	Mat imgHist(hist_rows, hist_cols, CV_8UC1, Scalar(255, 255, 255)); // constructs a white image
	
	int baseline = hist_rows - 1;
	for (int x = 0; x < hist_cols; x++) {
		printf("%d: %d\n", x, hist[x]);
		Point p1 = Point(x, baseline);
		Point p2 = Point(x, baseline - hist[x]);
		line(imgHist, p1, p2, Scalar(0, 0, 0)); // histogram bins colored in black
	}
	imshow(name, imgHist);
}

int* horizontalHistogram(Mat src) {
	int* hist_vect = (int*)calloc(src.rows, sizeof(int)); /*fiecare valoare din acest vector
														    retine, pentru fiecare rand, numarul
															de pixeli negri*/
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == BLACK) {
				hist_vect[i]++;
			}
		}
	}
	return hist_vect;
}

int* verticalHistogram(Mat src) {
	int* hist_vect = (int*)calloc(src.cols, sizeof(int)); /*fiecare valoare din acest vector
															retine, pentru fiecare coloana, numarul
															de pixeli negri*/
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			if (src.at<uchar>(i, j) == BLACK) {
				hist_vect[j]++;
			}
		}
	}
	return hist_vect;
}

void load_templates() {
	Mat sn1 = imread("Images/templates/sn1.jpg", IMREAD_GRAYSCALE);
	Mat sn2 = imread("Images/templates/sn2.jpg", IMREAD_GRAYSCALE);
	Mat sn3 = imread("Images/templates/sn3.jpg", IMREAD_GRAYSCALE);
	Mat sn4 = imread("Images/templates/sn4.jpg", IMREAD_GRAYSCALE);
	Mat sn5 = imread("Images/templates/sn5.jpg", IMREAD_GRAYSCALE);
	Mat sn6 = imread("Images/templates/sn6.jpg", IMREAD_GRAYSCALE);

	Mat en1 = imread("Images/templates/en1.jpg", IMREAD_GRAYSCALE);
	Mat en2 = imread("Images/templates/en2.jpg", IMREAD_GRAYSCALE);
	Mat en3 = imread("Images/templates/en3.jpg", IMREAD_GRAYSCALE);
	Mat en4 = imread("Images/templates/en4.jpg", IMREAD_GRAYSCALE);
	Mat en5 = imread("Images/templates/en5.jpg", IMREAD_GRAYSCALE);
	Mat en6 = imread("Images/templates/en6.jpg", IMREAD_GRAYSCALE);

	Mat cs1 = imread("Images/templates/cs1.jpg", IMREAD_GRAYSCALE);
	Mat cs2 = imread("Images/templates/cs2.jpg", IMREAD_GRAYSCALE);

	Mat bp1 = imread("Images/templates/bp1.jpg", IMREAD_GRAYSCALE);
	Mat bp2 = imread("Images/templates/bp2.jpg", IMREAD_GRAYSCALE);
	Mat bp3 = imread("Images/templates/bp3.jpg", IMREAD_GRAYSCALE);

	Mat mp1 = imread("Images/templates/mp1.jpg", IMREAD_GRAYSCALE);


	Mat dz1 = imread("Images/templates/dz1.jpg", IMREAD_GRAYSCALE);
	Mat dz2 = imread("Images/templates/dz2.jpg", IMREAD_GRAYSCALE);


	templates.push_back({ sn1, "solid" });
	//templates.push_back({ sn2, "solid" });
	//templates.push_back({ sn3, "solid" });
	//templates.push_back({ sn4, "solid" });
	//templates.push_back({ sn5, "solid" });
	//templates.push_back({ sn6, "solid" });

	templates.push_back({ en1, "empty" });
	//templates.push_back({ en2, "empty" });
	//templates.push_back({ en3, "empty" });
	//templates.push_back({ en4, "empty" });
	//templates.push_back({ en5, "empty" });
	//templates.push_back({ en6, "empty" });

	templates.push_back({ cs1, "clef" });
	//templates.push_back({ cs2, "clef" });

	//templates.push_back({ dz1, "sharp" });
	//templates.push_back({ dz2, "sharp" });

	templates.push_back({ bp2, "rest" });

	//templates.push_back({ bp1, "line" });
	templates.push_back({ bp3, "end line" });

	templates.push_back({ mp1, "masura" });
}

vector<dimensiuni_portativ> calcul_dimensiuni_portative(int* histograma_vector, int lungime_vector) {
	vector<dimensiuni_portativ> portative;
	int offset = 0, inaltime = 0;
	for (int i = 1; i < lungime_vector; i++) {
		if (histograma_vector[i - 1] == 0 && histograma_vector[i] == 0) { //prev e alb si eu sunt alb
			;								 //nu fac nimic, sunt in afara portativului
		}
		if (histograma_vector[i - 1] == 0 && histograma_vector[i] != 0) { //prev e alb si eu sunt black
			offset = i;						 //incepe portativ
		}
		if (histograma_vector[i - 1] != 0 && histograma_vector[i] != 0) { //prev e black si eu sunt black
			;								 //nu fac nimic, sunt in portativ
		}
		if (histograma_vector[i - 1] != 0 && histograma_vector[i] == 0) { //prev e black si eu sunt alb
			inaltime = i - offset;			 //termina portativ
			if (inaltime > 70) { // ca sa nu salveze 6 si 11 de deasupra cheii sol
				portative.push_back({ offset, inaltime });
			}
			offset = 0;
			inaltime = 0;
		}
	}
	return portative;
}

int creare_portative(Mat src, vector<dimensiuni_portativ> port) {
	int i = 0;
	for each(dimensiuni_portativ dim in port) {
		Mat myCopy = src.clone();
		Rect myROI(0, dim.yOffset - 35, myCopy.cols, dim.inaltime + 35); //marja de eroare bounds up & down
		Mat croppedImage = myCopy(myROI); // taie imaginea dupa myROI bounds
		char fname[20] = "portativ";
		char* increment = new char[1];
		_itoa(i, increment, 10);
		strcat(fname, increment);
		strcat(fname, ".jpg");
		imwrite(fname, croppedImage); //writes the destination to file
		portativeImagine.push_back(croppedImage);
		printf("[INFO] Creare portativ %d: %d x %d\n", i, croppedImage.rows, croppedImage.cols);
		i++;
	}
	return i;
}

Mat eliminare_linii_portativ(Mat sursa, int index) {
	Mat src = sursa.clone();
	int* vals = horizontalHistogram(src);
	int medie = 0;
	for (int i = 0; i < src.rows; i++) {
		medie += vals[i];
	}
	medie = medie / src.rows; //media valorilor histogramei
	printf("[INFO] Eliminare linii portativ: %d x %d\n", src.rows, src.cols);
	for (int i = 0; i < src.rows; i++) {
		if (vals[i] > 4 * medie) { //portativ aprox de 4 ori mai mare decat media
			for (int j = 0; j <src.cols ; j++) {
				if (src.at<uchar>(i-2, j) == WHITE || src.at<uchar>(i + 2, j) == WHITE) { /*pentru notele care se suprapun peste linia de 
																						  portativ; daca suntem in interiorul unei note inseamna
																						  ca deasupra si dedesubt avem pixeli negri; altfel, portativ*/
					src.at<uchar>(i, j) = WHITE; //sterge pixel 
				}
			}
		}
	}	
	char fname[20] = "el";
	char* increment = new char[1];
	_itoa(index, increment, 10);
	strcat(fname, increment);
	strcat(fname, ".jpg");
	imwrite(fname, src);
	return src;
}

void MatchingMethod(int, void*);

void metoda_template() {
	MatchingMethod(0, 0);
}

bool compareByScore(MatchedNoteWithScore a, MatchedNoteWithScore b) {
	return a.score < b.score;
}

bool contains(vector<MatchedNote> notes, MatchedNote noteToFind) {
	for each(MatchedNote n in notes) {
		if (n.p.x == noteToFind.p.x && n.p.y == noteToFind.p.y) {
			return true;
		}
	}
	return false;
}

void MatchingMethod(int, void*)
{
	for (int i = 0; i < 3; i++) {
		Mat portativ = portativeFitrate.at(i);
		vector<Subset> bounds = areas[i];
		printf("%d bounds: %d\n", i, bounds.size());

		for each(Subset s in bounds) {
			Range rows(0, portativ.rows-1);
			Range cols(s.start, s.end);
			Mat subImg = portativ(rows, cols);
			///TODO
			vector<MatchedNoteWithScore> findBestMatch;
			for each (TemplateType t in templates) {
				if (subImg.cols >= t.templ.cols) {
					matchTemplate(subImg, t.templ, result, TM_CCORR_NORMED);
					//normalize(result, result, 0, 1, NORM_MINMAX, -1, Mat());
					double minVal; double maxVal; Point minLoc; Point maxLoc;
					Point matchLoc;
					minMaxLoc(result, &minVal, &maxVal, &minLoc, &maxLoc, Mat());
					matchLoc = maxLoc;// minLoc;
					//notes[i].push_back({ matchLoc, t.type });
					//printf("maxVal: %f\n", maxVal);
					MatchedNote n = {matchLoc, t.type};
					findBestMatch.push_back({n, maxVal});
					//printf("size in loop: %d\n", findBestMatch.size());
				}
			}
			//am facut match cu toate template-urile pt zona aleasa, trebuie sa vad care e cel mai bun
			if (findBestMatch.size() > 0) {
				sort(findBestMatch.begin(), findBestMatch.end(), compareByScore);
				MatchedNote bestNote = findBestMatch.back().note;
				if (bestNote.p.y > 31) {
					if (!contains(notes[i], bestNote)) {
						notes[i].push_back(bestNote);
					}
				}
			}
			//printf("size: %d\n", findBestMatch.size());

		}
	}
	return;
}

Mat filtruMedian(Mat src, int index) { 
	printf("[INFO] Filtrare imagini portativ: %d x %d\n", src.rows, src.cols);
	Mat dst = src.clone();
	int k = 1;
	vector<uchar> vec;
	for (int i = 0; i < src.rows; i++) {
		for (int j = 0; j < src.cols; j++) {
			vec.clear();
			for (int ll = -k; ll <= k; ll++) {
				for (int lc = -k; lc <= k; lc++) {
					if (isInside(src, i + ll, j + lc))
						vec.push_back(src.at<uchar>(i + ll, j + lc));
				}
			}
			sort(vec.begin(), vec.end());
			dst.at<uchar>(i, j) = vec[vec.size() / 2];
		}
	}
	char fname[20] = "f";
	char* increment = new char[1];
	_itoa(index, increment, 10);
	strcat(fname, increment);
	strcat(fname, ".jpg");
	imwrite(fname, dst);
	return dst;
}

void prelucrare_imagine_sursa() {
	Mat src;
	src = imread("Images/binarized.jpg", CV_LOAD_IMAGE_GRAYSCALE);
	printf("[INFO] Dimensiuni imagine initiala: %d x %d\n", src.rows, src.cols);

	int* vals = horizontalHistogram(src);
	vector<dimensiuni_portativ> port = calcul_dimensiuni_portative(vals, src.rows); /*functie care determina dimensiunile fiecarui 
																					portativ, pe baza histogramei orizontale*/
	nrPortative = creare_portative(src, port);  //creeaza obiecte de tip Mat corespunzatoare fiecarui portativ
	printf("[INFO] Nr portative gasite in imagine: %d\n", nrPortative-1); //-1 pentru ca avem si titlul
	portativeImagine.pop_front(); //prima imagine e titlul melodiei
	portativeImagine.front();
	int index = 0;
	for each(Mat p in portativeImagine) {
		Mat empty = eliminare_linii_portativ(p, index); //functie care elimina liniile portativului
		portativeEmpty.push_back(empty);
		Mat filtered = filtruMedian(empty, index); /*functie care filtreaza imaginile (Mat-urile) 
												   corespunzatoare portativelor - elimina zgomote*/
		portativeFitrate.push_back(filtered);
		index++;
	}
}

bool compareByX(MatchedNote a, MatchedNote b) {
	return a.p.x < b.p.x;
}
bool compareByY(MatchedNote a, MatchedNote b) {
	return a.p.y < b.p.y;
}

void interest_areas() { //det unde sunt note in portativ
	vector<Subset> subsets; //stocheaza portiunile din fiecare portativ
	Subset subset;
	int i = 0;
	for each (Mat port in portativeFitrate) {
		int* histograma_vector = verticalHistogram(port);
		int lungime_vector = port.cols;

		int offset = 0, dim = 0; //offsetul la care incepe o nota, dimensiunea sa (cati pixeli ocupa)

		for (int i = 1; i < lungime_vector; i++) {

			if (histograma_vector[i - 1] == 0 && histograma_vector[i] == 0) { //prev e alb si eu sunt alb
				;								 //nu fac nimic, sunt in afara notei
			}
			if (histograma_vector[i - 1] == 0 && histograma_vector[i] != 0) { //prev e alb si eu sunt black
				offset = i;						 //incepe o nota
			}
			if (histograma_vector[i - 1] != 0 && histograma_vector[i] != 0) { //prev e black si eu sunt black
				;								 //nu fac nimic, sunt in nota
			}
			if (histograma_vector[i - 1] != 0 && histograma_vector[i] == 0) { //prev e black si eu sunt alb
				dim = i - offset;				 //termina nota
				//if (dim > 20) {
					subset.start = offset - WHITE_GAP;
					subset.end = offset + dim + WHITE_GAP;
					subsets.push_back(subset);
				//}
				
				offset = 0;
				dim = 0;
			}
		}
		areas[i] = subsets;
		i++;
	}
}

int main()
{
	load_templates();
	prelucrare_imagine_sursa();
	interest_areas();
	
	metoda_template();

	for (int i = 0; i < 3; i++) {
		//sort(notes[i].begin(), notes[i].end(), compareByX);
		sort(notes[i].begin(), notes[i].end(), compareByY);
		for each (MatchedNote n in notes[i]) {
			printf("%d: %s pos: x=%d, y=%d\n", i, n.noteName, n.p.x, n.p.y);
		}
	}
	
	return 0;
}