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
 */

void movePlayer(character * player, bool key[])
{
	//Move player
	if(key[KEY_UP]) {
		player->y -= player->y_speed;
	}
	if(key[KEY_DOWN]) {
		player->y += player->y_speed;
	}
	if(key[KEY_LEFT]) {
		player->x -= player->x_speed;
	}
	if(key[KEY_RIGHT]) {
		player->x += player->x_speed;
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

	//player_bitmap = al_create_bitmap(PLAYER_SIZE, PLAYER_SIZE);
	//enemy_bitmap = al_create_bitmap(ENEMY_SIZE, ENEMY_SIZE);


        al_init_image_addon();
	player_bitmap = al_load_bitmap(BITMAP_PLAYER_PATH);
	enemy_bitmap = al_load_bitmap(BITMAP_ENEMY_PATH);
	
	if(!player_bitmap || !enemy_bitmap) {
		fprintf(stderr, "failed to create bitmap!\n");
		return -1;
	}

	/*** Initialize game variables ***/
	initPlayer(&player);
	initEnemy(&enemy);
	score = 0;
	highscore = 0;

	/*** Register events ***/
	al_register_event_source(event_queue, al_get_display_event_source(display));
	al_register_event_source(event_queue, al_get_timer_event_source(timer));
	al_register_event_source(event_queue, al_get_timer_event_source(speed_timer));

	/*** Create bitmaps ***/
	//al_set_target_bitmap(player_bitmap);
	//al_clear_to_color(al_map_rgb(0, 0, 255));

	//al_set_target_bitmap(enemy_bitmap);
	//al_clear_to_color(al_map_rgb(255, 0, 0));

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
				movePlayer(&player, keys);
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
		else if (ev.type == ALLEGRO_EVENT_KEY_UP)
		{
		    keys[KEY_UP] = true;
		}	

		else if (ev.type == ALLEGRO_EVENT_KEY_DOWN)
		{
		    keys[KEY_DOWN] = true;
		}	

		else if (ev.type == ALLEGRO_KEY_LEFT)
		{
		    keys[KEY_LEFT] = true;
		}	

		else if (ev.type == ALLEGRO_KEY_RIGHT)
		{
		    keys[KEY_RIGHT] = true;
		}	
 
	  	/*** Update frame ***/
	  	if(redraw && al_is_event_queue_empty(event_queue)) {

			//Redraw and flip the display
			//Note: Allegro draws to an empty "buffer" display, which is show once al_flip_display() is called.
			redraw = false;
			al_clear_to_color(al_map_rgb(255,255,255));
			al_draw_bitmap(player_bitmap, player.x, player.y, 0);
			al_draw_bitmap(enemy_bitmap, enemy.x, enemy.y, 0);
			al_flip_display();

		 	/*** Game over ***/
		 	if(checkCollision(player, enemy)) {
				al_stop_timer(timer);
				al_rest(3);
				al_start_timer(timer);

                al_install_audio();
				al_reserve_samples(1);
				ALLEGRO_SAMPLE* ptrSample = al_load_sample(AUDIO_SAMPLE_PATH);
				if (ptrSample != NULL)
				{
	                al_play_sample(ptrSample, 1.0, 0.0, 1.0, ALLEGRO_PLAYMODE_LOOP, NULL);				   	
				}	


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