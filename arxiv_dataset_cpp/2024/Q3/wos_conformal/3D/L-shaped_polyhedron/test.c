
#define RUN_TEST(test) \
	do { \
		printf("Running " #test "... "); \
		if (test()) printf("[OK]"); \
		printf("\n"); \
	} while(0)

#define VERIFY_TRUE(condition, remark) \
	do { \
		if (!(condition)) { \
			printf("\n  Line %d : " #condition " failed.", __LINE__); \
			printf("%s", remark); \
			return false; \
		} \
	} while(0)

#define VERIFY_POINT_VALID(point) \
	do { \
		VERIFY_TRUE(!(point->y == 0 && point->z == 0), ""); \
		VERIFY_TRUE(!(point->y == 1 && point->z == 0), ""); \
		VERIFY_TRUE(!(point->x == 0 && point->y == 0), ""); \
		VERIFY_TRUE(!(point->x == 0 && point->y == 1), ""); \
		VERIFY_TRUE(!(point->x == 0 && point->z == 0), ""); \
		VERIFY_TRUE(!(point->x == 1 && point->z == 1), ""); \
		VERIFY_TRUE(!(point->x >= 1 && point->x <= 3 && point->y == 0 && point->z == 1), ""); \
		VERIFY_TRUE(!(point->x >= 1 && point->x <= 3 && point->y == 1 && point->z == 1), ""); \
		VERIFY_TRUE(!(point->z >= 1 && point->z <= 2 && point->y == 0 && point->x == 1), ""); \
		VERIFY_TRUE(!(point->z >= 1 && point->z <= 2 && point->y == 1 && point->x == 1), ""); \
	} while(0)

static bool test_get_points() {
	printf("(%03d, %03d, %03d)", X_SLIDES, Y_SLIDES, Z_SLIDES);
	int counter;
	Point* list = get_points(&counter);
	VERIFY_TRUE(counter > 0, "Number of points generated should be positive");
	for (int i = 0; i < counter; i++) {
		VERIFY_TRUE(list[i]!=NULL, "Missing points");
	}
	for (int i = 0; i < counter; i++) {
		VERIFY_POINT_VALID((list[i]));
	}
	return true;
}
static bool test_find_min_radius() {
	double r = 0;
	Point pt = create_point(1.6, 0.5, 1.0);
	r = find_min_radius(pt);
	VERIFY_TRUE(r == 0.5, "");
	free(pt);

	pt = create_point(1, 0, 0.6);
        r = find_min_radius(pt);
        VERIFY_TRUE(r == 0.4, "");
	free(pt);

	pt = create_point(0.5, 0.6, 1.5);
        r = find_min_radius(pt);
        VERIFY_TRUE(r == sqrt(0.41), "");
	free(pt);

	pt = create_point(0.5, 0.4, 0.5);
        r = find_min_radius(pt);
        VERIFY_TRUE(r == sqrt(0.41), "");
        free(pt);

	return true;
}

static bool test_move_to_next() {
	Point pt = create_point(0, 0, 0);
	move_to_next(pt, 1);
	double r = (pt->x)*(pt->x) + (pt->y)*(pt->y) + (pt->z)*(pt->z);
	VERIFY_TRUE(r >= 1-1e-12 && r <= 1+1e-12, "");
	return true;
}
static bool test_bounce_inside() {
	Point pt = create_point(1, -0.5, 0.432);
	bounce_inside(pt);
	VERIFY_TRUE(pt->x == 1 && pt->y == 0.5 && pt->z == 0.432, "");
	free(pt);

	pt = create_point(1.2, 0.5, 1.1);
        bounce_inside(pt);
        VERIFY_TRUE(pt->x == 1.2 && pt->y == 0.5 && pt->z+1 == 1.9, "");
	free(pt);

	pt = create_point(0.6742, 1.672, 0.8);
        bounce_inside(pt);
        VERIFY_TRUE(pt->x == 0.6742 && pt->y+1 == 1.328 && pt->z == .8, "");
	free(pt);
	return true;
}
static bool test_isOnPlanes() {
	int mark = 0;
	Point pt = create_point(0.5, 0.5, 2);
	VERIFY_TRUE(isOnPlanes(pt, &mark), "");
	VERIFY_TRUE(mark == 0, "");
	free(pt);
	
	pt = create_point(3, 0.5, 0.5);
        VERIFY_TRUE(isOnPlanes(pt, &mark), "");
        VERIFY_TRUE(mark == 1, "");
        free(pt);

	pt = create_point(0.5+PRECISE, 0.5, 2);
        VERIFY_TRUE(isOnPlanes(pt, &mark), "");
        VERIFY_TRUE(mark == 1, "");
        free(pt);

	pt = create_point(3-PRECISE, 0.5, 0.5);
        VERIFY_TRUE(isOnPlanes(pt, &mark), "");
        VERIFY_TRUE(mark == 2, "");
        free(pt);

	return true;
}
int main() {
	RUN_TEST(test_get_points);
	RUN_TEST(test_find_min_radius);
	RUN_TEST(test_move_to_next);
	RUN_TEST(test_bounce_inside);
	RUN_TEST(test_isOnPlanes);
	return 0;
}
