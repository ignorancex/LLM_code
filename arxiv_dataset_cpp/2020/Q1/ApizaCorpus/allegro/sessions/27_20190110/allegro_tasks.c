#include <stdio.h>
#include <stdlib.h>
#include <allegro5/allegro.h>
#include <allegro5/allegro_image.h>
#include <allegro5/allegro_audio.h>
#include <allegro5/allegro_acodec.h>
#include <allegro5/allegro_font.h>
#include <allegro5/allegro_ttf.h>
 
/*
 *    Title: Spider and Fly
 *
 *    Description:
 *    "Spider and Fly" is a simple game written in C using the Allegro library, in which the player
 *    uses the arrow keys to move the fly and avoid colliding with the spider. Every 5 seconds, both
 *    characters speeds increase. When the two collide, the player's score is set to 0 and
 *    both characters reset to their starting positions and speeds.
 *
 *    Many basic features have already been implemented. You must work to implement the remaining features
 *    described in the readme file.
 *
 *
 */

/****** Global variables *******/ 
//NOTE: You should not need to modify the existing global variables, though you may add new ones
//If you are having frame-rate issues, you may want to decrease the FPS.

typedef struct character character;

struct character {
   int   size;
   int   x;
   int   y;
   int   x_speed;
   int   y_speed;
};

const char * AUDIO_SAMPLE_PATH = "assets/sample.wav"; 
const char * BITMAP_PLAYER_PATH = "assets/fly.png"; 
const char * BITMAP_ENEMY_PATH = "assets/spider.png";
const char * FONT_PATH = "assets/roboto.ttf"; 

const float FPS = 24;
const int SCREEN_W = 640;
const int SCREEN_H = 480;
const int PLAYER_SIZE = 32;
const int ENEMY_SIZE = 96;
const int DEFAULT_SPEED = 5;
const int MAX_SPEED = 100;
const int ENEMY_SPEEDUP_RATE = 5;

enum MYKEYS {
	KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT
};



/********* Helper Functions ***********/
//NOTE: You should not need to modify the existing helper functions, though you may add new ones

/*    
 *    Function: initPlayer
 *    Initializes the player character at its starting position with default speed
 */

void initPlayer(character * player)
{
	player->size = PLAYER_SIZE;
	player->x = SCREEN_W / 4.0 - PLAYER_SIZE / 2.0;
	player->y = SCREEN_H / 2.0 - PLAYER_SIZE / 2.0;
	player->x_speed = DEFAULT_SPEED;
	player->y_speed = DEFAULT_SPEED;
}


/*    
 *    Function: initEnemy
 *    Initializes the enemy character at its starting position with random speed
 */

void initEnemy(character * enemy)
{
	enemy->size = ENEMY_SIZE;
	enemy->x = 3 * SCREEN_W / 4.0 - ENEMY_SIZE / 2.0;
	enemy->y = SCREEN_H / 2.0 - ENEMY_SIZE / 2.0;
	enemy->x_speed = rand()%(DEFAULT_SPEED)+2;
	enemy->y_speed = rand()%(DEFAULT_SPEED)+2;
}


/*    
 *    Function: movePlayer
 *    Varies the player character's x and y coordinates according to its speed 
 *    and a bool array representing the keys currently pressed.
 *    Only moves the character if it won't be moved beyond the screen.
 *    Returns the angle (in radians) the player is moving, or -10000.0 if the player is not moving.
 */

float movePlayer(character * player, bool key[])
{
	float angle = -10000.0;
	//Move player
	if(key[KEY_UP]) {
		player->y -= player->y_speed;
		angle = 0.0;
	}
	if(key[KEY_DOWN]) {
		player->y += player->y_speed;
		angle = 3.141592653;
	}
	if(key[KEY_LEFT]) {
		player->x -= player->x_speed;
		angle = -1.57079632679;
	}
	if(key[KEY_RIGHT]) {
		player->x += player->x_speed;
		angle = 1.57079632679;
	}

	//Correct position if out of bounds
	if(player->y < 0) {
		player->y = 0;
	}else if (player->y > SCREEN_H - player->size) {
		player->y = SCREEN_H - player->size;
	}
	if(player->x < 0) {
		player->x = 0;
	}else if (player->x > SCREEN_W - player->size) {
		player->x = SCREEN_W - player->size;
	}
	return angle;
}


/*    
 *    Function: moveEnemy
 *    Varies the enemy character's x and y coordinates according to its speed 
 *    If the enemy hits the edge of the window, it will bounce back in the opposite direction
 */

void moveEnemy(character * enemy){
	//Move enemy
	enemy->y += enemy->y_speed;
	enemy->x += enemy->x_speed;

	//Correct position if out of bounds
	if(enemy->y < 0) {
		enemy->y = 0;
		enemy->y_speed *= -1;
	}else if (enemy->y > SCREEN_H - enemy->size) {
		enemy->y = SCREEN_H - enemy->size;
		enemy->y_speed *= -1;
	}
	if(enemy->x < 0) {
		enemy->x = 0;
		enemy->x_speed *= -1;
	}else if (enemy->x > SCREEN_W - enemy->size) {
		enemy->x = SCREEN_W - enemy->size;
		enemy->x_speed *= -1;
	}
}


/*    
 *    Function: speedUp
 *    Increases the character's x and y speeds by a factor of 1.5, as long as both are below the MAX_SPEED
 */

void speedUp(character * c) {
	if(c->y_speed < MAX_SPEED && c-> x_speed < MAX_SPEED) {
		c->y_speed *= 1.5;
		c->x_speed *= 1.5;
	}
}


/*    
 *    Function: checkCollision
 *    Checks for a (hitbox) collision between the characters c1 and c2
 *    Return 1 if a collision is detected, 0 otherwise 
 */

bool checkCollision(character c1, character c2) {
	if ((c1.x > c2.x + c2.size - 1) ||
		(c1.y > c2.y + c2.size - 1) ||
		(c2.x > c1.x + c1.size - 1) || 
		(c2.y > c1.y + c1.size - 1)) {
		// No collision
		return 0;
   	}
	// Collision
	return 1;
}



/************ Main ***************/
//Your new features will be implemented here

int main(int argc, char **argv){

	ALLEGRO_DISPLAY *display;
	ALLEGRO_EVENT_QUEUE *event_queue;
	ALLEGRO_TIMER *timer, *speed_timer;
	ALLEGRO_BITMAP *player_bitmap, *enemy_bitmap;
	ALLEGRO_BITMAP *image_bitmap;
	ALLEGRO_SAMPLE *game_over_spl;
	ALLEGRO_FONT *scores_font;
	char scores_str[32];
	int sw, sh;
	float last_heading = 0.0;
	bool paused = false;

	character player, enemy;
	int score, highscore;

	bool keys[4] = { false, false, false, false };
	bool redraw = true;
	bool doexit = false;

	/*** Initialization ***/
	if(!al_init()) {
		fprintf(stderr, "failed to initialize allegro!\n");
		return -1;
	}

	display = al_create_display(SCREEN_W, SCREEN_H);
	if(!display) {
		fprintf(stderr, "failed to create display!\n");
		return -1;
	}

	event_queue = al_create_event_queue();
	if(!event_queue) {
		fprintf(stderr, "failed to create event_queue!\n");
		return -1;
	}

	timer = al_create_timer(1.0 / FPS);
	speed_timer = al_create_timer(ENEMY_SPEEDUP_RATE);
	if(!timer || !speed_timer) {
		fprintf(stderr, "failed to create timer!\n");
		return -1;
	}

	player_bitmap = al_create_bitmap(PLAYER_SIZE, PLAYER_SIZE);
	enemy_bitmap = al_create_bitmap(ENEMY_SIZE, ENEMY_SIZE);
	if(!player_bitmap || !enemy_bitmap) {
		fprintf(stderr, "failed to create bitmap!\n");
		return -1;
	}

	/*** Initialize audio ***/
	al_install_audio();
	al_reserve_samples(1);
	al_init_acodec_addon();
	game_over_spl = al_load_sample(AUDIO_SAMPLE_PATH);
	if(game_over_spl == NULL) {
		fprintf(stderr, "failed to load audio!\n");
		return -1;
	}

	al_init_font_addon();
	al_init_ttf_addon();
	scores_font = al_load_font(FONT_PATH, 36, 0);
	if(scores_font == NULL) {
		fprintf(stderr, "failed to load font!\n");
		return -1;
	}

	al_install_keyboard();
	al_install_mouse();

	/*** Initialize game variables ***/
	initPlayer(&player);
	initEnemy(&enemy);
	score = 0;
	highscore = 0;

	/*** Register events ***/
	al_register_event_source(event_queue, al_get_display_event_source(display));
	al_register_event_source(event_queue, al_get_timer_event_source(timer));
	al_register_event_source(event_queue, al_get_timer_event_source(speed_timer));
	al_register_event_source(event_queue, al_get_keyboard_event_source());
	al_register_event_source(event_queue, al_get_mouse_event_source());

	/*** Create bitmaps ***/
	al_init_image_addon();
	al_set_target_bitmap(player_bitmap);
	al_clear_to_color(al_map_rgb(255, 255, 255));
	image_bitmap = al_load_bitmap(BITMAP_PLAYER_PATH);
	sw = al_get_bitmap_width(image_bitmap);
	sh = al_get_bitmap_height(image_bitmap);
	al_draw_scaled_bitmap(image_bitmap, 0, 0, sw, sh, 0, 0, 32, 32, 0);

	al_set_target_bitmap(enemy_bitmap);
	al_clear_to_color(al_map_rgb(255, 255, 255));
	image_bitmap = al_load_bitmap(BITMAP_ENEMY_PATH);
	sw = al_get_bitmap_width(image_bitmap);
	sh = al_get_bitmap_height(image_bitmap);
	al_draw_scaled_bitmap(image_bitmap, 0, 0, sw, sh, 0, 0, 96, 96, 0);

	al_set_target_bitmap(al_get_backbuffer(display));
	al_clear_to_color(al_map_rgb(255,255,255));
	al_flip_display();

	/*** Start timers ***/
	al_start_timer(timer);
	al_start_timer(speed_timer);


	/*** Game loop ***/
	while(!doexit) {
		ALLEGRO_EVENT ev;
		al_wait_for_event(event_queue, &ev);

		/*** Timer event ***/
		if(ev.type == ALLEGRO_EVENT_TIMER) {
			if(ev.timer.source == timer) {
				redraw = true;
				float new_heading = movePlayer(&player, keys);
				if(new_heading > -10000.0){
					last_heading = new_heading;
				}
				moveEnemy(&enemy);
				score ++;
				if(score > highscore) highscore = score;
			}
			else if(ev.timer.source == speed_timer) {
				speedUp(&enemy);
				speedUp(&player);
			}
		}

		/*** Display close event ***/
		else if(ev.type == ALLEGRO_EVENT_DISPLAY_CLOSE) {
			break;
		}

		/*** Keyboard event ***/
		else if(ev.type == ALLEGRO_EVENT_KEY_DOWN) {
			switch(ev.keyboard.keycode) {
			case ALLEGRO_KEY_LEFT:
				keys[KEY_LEFT] = true;
				break;
			case ALLEGRO_KEY_RIGHT:
				keys[KEY_RIGHT] = true;
				break;
			case ALLEGRO_KEY_UP:
				keys[KEY_UP] = true;
				break;
			case ALLEGRO_KEY_DOWN:
				keys[KEY_DOWN] = true;
				break;
			}
		}
		else if(ev.type == ALLEGRO_EVENT_KEY_UP) {
			switch(ev.keyboard.keycode) {
			case ALLEGRO_KEY_LEFT:
				keys[KEY_LEFT] = false;
				break;
			case ALLEGRO_KEY_RIGHT:
				keys[KEY_RIGHT] = false;
				break;
			case ALLEGRO_KEY_UP:
				keys[KEY_UP] = false;
				break;
			case ALLEGRO_KEY_DOWN:
				keys[KEY_DOWN] = false;
				break;
			}
		}

		/*** Mouse event ***/
		else if(ev.type == ALLEGRO_EVENT_MOUSE_BUTTON_DOWN) {
			paused = !paused;
			if(paused) {
				al_stop_timer(timer);
				al_stop_timer(speed_timer);
			}else{
				al_start_timer(timer);
				al_start_timer(speed_timer);
			}
		}
 
	  	/*** Update frame ***/
	  	if(redraw && al_is_event_queue_empty(event_queue)) {

			//Redraw and flip the display
			//Note: Allegro draws to an empty "buffer" display, which is show once al_flip_display() is called.
			redraw = false;
			al_clear_to_color(al_map_rgb(255,255,255));
			al_draw_rotated_bitmap(player_bitmap, 16, 16, player.x, player.y, last_heading, 0);
			al_draw_bitmap(enemy_bitmap, enemy.x, enemy.y, 0);
			sprintf(scores_str, "%d", score);
			al_draw_text(scores_font, al_map_rgb(0,0,0), 20, 20, ALLEGRO_ALIGN_LEFT, scores_str);
			sprintf(scores_str, "%d", highscore);
			// TODO get window width and use that to determine x
			al_draw_text(scores_font, al_map_rgb(0,0,0), SCREEN_W - 20, 20, ALLEGRO_ALIGN_RIGHT, scores_str);
			al_flip_display();

		 	/*** Game over ***/
		 	if(checkCollision(player, enemy)) {
				al_play_sample(game_over_spl, 1.0, 0.0, 1.0, ALLEGRO_PLAYMODE_ONCE, NULL);
				al_stop_timer(timer);
				al_rest(3);
				al_start_timer(timer);
				initPlayer(&player);
				initEnemy(&enemy);
				score = 0;
 			}
		}
	}		
 
	/*** Clean up ***/
	al_destroy_bitmap(player_bitmap);
	al_destroy_bitmap(enemy_bitmap);
	al_destroy_timer(timer);
	al_destroy_display(display);
	al_destroy_event_queue(event_queue);

	return 0;
}