/* >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
   bwt-based compression and indexing
   Basic routines for uncompressing the compressed index
   decompr_main.c   ------  P. Ferragina & G. Manzini, 10 June 2000

     decompress_file();
     read_prologue();
     read_basic_prologue();
     uncompress_data();
     uncompress_superbucket();
     compute_bwt_occ();
     compute_lf();
     invert_bwt();
     write_text();
     get_char_maps();
   >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> */
#include "common.h"




/* *********************************************************
   read basic prologue of a .bwi file
   ********************************************************* */
void read_basic_prologue(FM_INDEX *Infile,bwi_out *s)
{
  void init_bit_buffer(FM_INDEX *);
  void fatal_error(char *);
  void out_of_mem(char *);



  //int uint_read(void);
  int uint_read(FM_INDEX *Infile);
  //int bit_read(FILE *,int);
  int bit_read(FM_INDEX *Infile,int n);
  //int my_fseek(FILE *f, long offset, int whence);
  int int_log2(int),int_pow2(int);
  int i,size;

  my_fseek(Infile,0,SEEK_SET);       // rewind
  init_bit_buffer(Infile);                 // initialize read-buffer

  Infile->Type_compression = bit_read(Infile,8);

  s->text_size = uint_read(Infile);
  s->bwt_eof_pos = uint_read(Infile);
  Infile->Bucket_size_lev1 = bit_read(Infile,16)<<10;
  Infile->Bucket_size_lev2 = bit_read(Infile,8)<<10;

  assert((Infile->Bucket_size_lev1 % Infile->Bucket_size_lev2) == 0);

  Infile->Num_bucs_lev1 = (s->text_size + Infile->Bucket_size_lev1 - 1)/Infile->Bucket_size_lev1;
  Infile->Num_bucs_lev2 = (s->text_size + Infile->Bucket_size_lev2 - 1)/Infile->Bucket_size_lev2;

  // ---- mtf & alphabet information
  Infile->Mtf_save = bit_read(Infile,8)+1;
  s->alpha_size = bit_read(Infile,8)+1;

  // ---- some information for the user --------
  if(Verbose) {
    fprintf(stderr,"Compression method: ");
    switch(Infile->Type_compression)
    {
    case ARITH:
      fatal_error("Arith. coding no longer available -read_basic_prologue-\n");
      exit(1);
    case HIER3:
      fprintf(stderr,"Hierarchical 3-level coding. "); break;
    case UNARY:
      fatal_error("Unary coding no longer available -read_basic_prologue-\n");
      exit(1);
    case MULTIH:
      fprintf(stderr,"Huffman with multiple tables. "); break;
    default:
      fprintf(stderr,"Error: invalid compression method\n"); exit(1);
    }
    fprintf(stderr,"Size of mtf list %d.\n",Infile->Mtf_save);
  }
  if(Infile->Type_compression==MULTIH && Infile->Mtf_save!=256)
    fatal_error("Invalid size of MTF list -read_basic_prologue-\n");

  // ---- read Chosen_char & starting position of occ list
  s->chosen_char = (uchar) bit_read(Infile,8);
  s->skip =  uint_read(Infile);
  Infile->Start_prologue_occ = uint_read(Infile);

  // ---- alphabet info and inverse char maps
  for(i=0;i<256;i++)
    s->bool_char_map[i] = bit_read(Infile,1);

  for(i=0,size=0;i<256;i++)
    if(s->bool_char_map[i]) {
      s->char_map[i]=size;
      s->inv_char_map[size++]= (uchar) i;
    }
  assert(size==s->alpha_size);

  // ---- prefix summed char-occ info
  for(i=0; i<s->alpha_size;i++)
    s->bwt_occ[i] = uint_read(Infile);

  // ---- compute other important infos on the file
  Infile->Start_prologue_info_sb = SIZE_BASIC_PROLOGUE;
  Infile->Start_prologue_info_sb += sizeof(int) * s->alpha_size;
  Infile->Start_prologue_info_b = Infile->Start_prologue_info_sb;
  Infile->Start_prologue_info_b += ((s->alpha_size+7)/8) * Infile->Num_bucs_lev1;
  Infile->Start_prologue_info_b += (Infile->Num_bucs_lev1-1) * sizeof(int)* s->alpha_size;

  //init the superbuckets' list
  s->buclist_lev1=malloc(Infile->Num_bucs_lev1*sizeof(bucket_lev1*)) ;
  if(s->buclist_lev1==NULL) out_of_mem("alloc_superbuckets");
  int sbi;
  for (sbi=0;sbi<Infile->Num_bucs_lev1;sbi++)
    		s->buclist_lev1[sbi]=NULL;

  //init the buckets' list
  s->buclist_lev2=malloc(Infile->Num_bucs_lev2*sizeof(bucket_lev2*)) ;
  if(s->buclist_lev2==NULL) out_of_mem("alloc_buckets");
  int bi;
  for (bi=0;bi<Infile->Num_bucs_lev2;bi++)
	  s->buclist_lev2[bi]=NULL;

}


/* *********************************************************
   read prologue of a .bwi file
   Input
     input file
   Output
     Bucket_size, s->text_size, s->bwt_eof_pos, s->bool_char_map
     s->alpha_size, s->numbucs, s->buclist
   ********************************************************* */
//DA NON USARE!!!!
void read_prologue(FM_INDEX *Infile,bwi_out *s)
{
  void out_of_mem(char *s);
  //int uint_read(void);
  int uint_read(FM_INDEX *Infile);
  int bit_read(FM_INDEX *Infile,int n);
  int int_log2(int);
  bucket_lev1 *sb;
  int i,rem,k,len;

  // ---- read basic infos from the prologue ----
  read_basic_prologue(Infile,s);

  // ----- alloc superbuckets ----------
  s->buclist_lev1=malloc(Infile->Num_bucs_lev1*sizeof(bucket_lev1*)) ;
  if(s->buclist_lev1==NULL) out_of_mem("alloc_superbuckets");
  int sbi;
  for (sbi=0;sbi<Infile->Num_bucs_lev1;sbi++)
  		s->buclist_lev1[sbi]=NULL;
  //s->buclist_lev1 = (bucket_lev1*)malloc(Infile->Num_bucs_lev1 * sizeof(bucket_lev1));


  // ------ alloc aux array for each superbucket ----
  for(i=0; i< Infile->Num_bucs_lev1; i++){
    sb = &s->buclist_lev1[i];

    // allocate space for array of occurrences
    sb->occ = (int *) malloc((s->alpha_size)* sizeof(int));
    if(sb->occ==NULL) out_of_mem("alloc_superbuckets");

    // allocate space for array of boolean char map
    sb->bool_char_map = (uchar *)malloc((s->alpha_size)*sizeof(uchar));
    if(sb->bool_char_map == NULL) out_of_mem("alloc_superbuckets");
  }

  // ----- init superbuckets ---
  if(s->alpha_size%8)
    rem =  8-(s->alpha_size % 8);  // required to keep data byte-alligned
  else
    rem = 0;

  for(i=0;i<Infile->Num_bucs_lev1;i++) {
    sb = &s->buclist_lev1[i];

    for(k=0; k<s->alpha_size; k++)      // ---- boolean char_map
      sb->bool_char_map[k]= bit_read(Infile,1);

    if(rem) bit_read(Infile,rem);             // byte alignment

    if(i>0)                            // read prefix-occ
      for(k=0;k<s->alpha_size;k++)
        sb->occ[k] = uint_read(Infile);
  }

  // ---- alloc array for the starting positions of the buckets -----
  s->start_lev2 =  (int *) malloc((Infile->Num_bucs_lev2)* sizeof(int));
  if(s->start_lev2==NULL) out_of_mem("alloc_buckets");

  // ----- read the start positions of the buckets -----
  len = (int_log2(s->text_size)+7)/8;   // variable length representation
  for(i=0;i<Infile->Num_bucs_lev2;i++)
    s->start_lev2[i] = bit_read(Infile,len*8);
}





/* ************************************************************
   expand the superbucket num. the uncompressed data is written
   in the array out[] which should be of the appropriate size
   (i.e. Bucket_size_lev1 unless num is the last superbucket)
   ************************************************************ */
void uncompress_superbucket(FM_INDEX *Infile,bwi_out *s, int num, uchar *out)
{
  int read7x8(FM_INDEX *Infile);
  void uncompress_bucket_hierarchy(uchar *, int, int);
  void uncompress_bucket_multihuf(uchar *, int, int);
  void fatal_error(char *);
  void init_bit_buffer(FM_INDEX *);
  bucket_lev1 *sb;
  uchar *dest, c, inv_char_map[256];
  int sb_start,sb_end,start,len;
  int b2,j,k;

  assert(num<Infile->Num_bucs_lev1);
  sb = &(s->buclist_lev1[num]);    // current superbucket
  sb_start=num*Infile->Bucket_size_lev1;   // starting position of superbucket
  sb_end = MIN(sb_start+Infile->Bucket_size_lev1,s->text_size);
  b2 = sb_start/Infile->Bucket_size_lev2;  // initial level 2 bucket

  sb->alpha_size=0;                // build inverse char map for superbucket
  for(k=0;k<s->alpha_size;k++)
    if(sb->bool_char_map[k])
      inv_char_map[sb->alpha_size++]=k;

  for(start=sb_start;start<sb_end;start+=Infile->Bucket_size_lev2,b2++) {
    if(fseek(Infile->file,s->start_lev2[b2],SEEK_SET)) // go to start of bucket
       fatal_error("fseek error -uncompress_superbucket");
    len = MIN(Infile->Bucket_size_lev2,sb_end-start); // length of bucket
    dest = out + (start-sb_start);

    init_bit_buffer(Infile);      // discard any leftout in the buffer
    if(start!=sb_start)     // if not the first bucket skip occ
      for(k=0;k<sb->alpha_size;k++)
         sb->occ[k]=read7x8(Infile);

    // -- Applies the proper decompression routine --
    switch (Infile->Type_compression)
      {
      case HIER3:  // ---- 3-level model: Fenwick's proposal -----
	uncompress_bucket_hierarchy(dest,len,sb->alpha_size);
	break;
      case MULTIH: // ---- Bzip compression of mtf-ranks -----
	uncompress_bucket_multihuf(dest,len,sb->alpha_size);
	break;
      case ARITH:
	fatal_error("Arith. no longer available -uncompress_superbucket-\n");
	exit(1);
      case UNARY:
	fatal_error("Unary no longer available -uncompress_superbucket-\n");
	exit(1);
      default:
	fprintf(stderr,"\n Compression algorithm unknown! ");
	fprintf(stderr,"-uncompress_superbucket-\n");
	exit(1);
      }

    // --- remap the bucket according to the superbucket inv_char_map ---
    for(j=0;j<len;j++) {                   // update occ[] and remap
      assert(dest[j]<sb->alpha_size);
      c=inv_char_map[dest[j]];             // compute remapped char
      assert(c<s->alpha_size);
      dest[j]=c;                           // remap
    }
  }
}

/* **********************************************************
   retrieve the bwt by uncompressing the data in the input file
   Input
     Infile, s->text_size
   Output
     s->bwt
   ********************************************************** */
void uncompress_data(FM_INDEX *Infile,bwi_out *s)
{
  //void uncompress_superbucket(bwi_out *, int, uchar *);
  void out_of_mem(char *s);
  double getTime( void );
  int i;

  s->bwt = (uchar *) malloc(s->text_size);
  if(s->bwt == NULL)
    out_of_mem("uncompress_data");

  if(Verbose)
    fprintf(stderr,"start decompression! (%.2f seconds)... ",getTime());
  for(i=0;i<Infile->Num_bucs_lev1;i++)
    uncompress_superbucket(Infile,s,i,s->bwt+i*Infile->Bucket_size_lev1);
  if(Verbose) fprintf(stderr,"done! (%.2f seconds)\n\n",getTime());
}




/* *************************************************************
   compute the lf mapping (see paper)
   Input
     s->bwt, s->bwt_occ[], s->text_size, s->alpha_size
   Output
     s->lf
   ************************************************************* */
void compute_lf(bwi_out *s)
{
  void out_of_mem(char *s);
  int i, occ_tmp[256];

  /* ------------ alloc memory ----------- */
  s->lf = (int *) malloc(s->text_size*sizeof(int));
  if(s->lf == NULL)
    out_of_mem("compute_lf");

  /* ----- copy bwt_occ ------ */
  for(i=0;i<256;i++)
    occ_tmp[i]=s->bwt_occ[i];

  /* ----- now computes lf mapping -------- */
  for(i=0;i<s->text_size;i++)
    s->lf[i] = occ_tmp[s->bwt[i]]++;
}


/* ***********************************************************
   compute the inverse bwt using the lf mapping
   Input
     s->bwt, s->bwt_eof_pos, s->text_size, s->lf
   Output
     s->text
   *********************************************************** */
void invert_bwt(bwi_out *s)
{
  void out_of_mem(char *s);
  int i,j;

  /* ------------ alloc memory ----------- */
  s->text = (uchar *) malloc(s->text_size);
  if(s->text == NULL)
    out_of_mem("invert_bwt");

  for(j=0,i=s->text_size-1;i>=0;i--) {
    s->text[i] = s->bwt[j];
    j = s->lf[j];              // No account for EOF

    assert(j>=0 && j<s->text_size);

    if(j<s->bwt_eof_pos) j++;  // EOF is not accounted in c[] and thus lf[]
                               // reflects the matrix without the first row.
                               // Since EOF is not coded, the lf[] is correct
                               // after bwt_eof_pos but it is -1 before.
                               // The ++ takes care of this situation.

  }
  assert(j==s->bwt_eof_pos);

}

/* ***********************************************
   write the uncompressed text to the output file
   Input
     s->text, s->inv_char_map, s->text_size
   Output
     output file
   *********************************************** */
void write_text(FILE *f, bwi_out *s)
{
  int i;


  for(i=0;i<s->text_size;i++)
    s->text[i] = s->inv_char_map[s->text[i]];

  fwrite(s->text,1,s->text_size,f);
}

















