library(readxl)
install.packages("sf")
library(sf)
install.packages('spdep')
library(spdep)
library(sp)
library(tictoc)
install.packages('lwgeom')
library(lwgeom)

tic()
setwd("/home/pranshu/Documents/Project Course/github1/c3.ai_minimalists/R_code/")
# Read input data
Link_distance   = read_xlsx("Data_input/route_final.xlsx") # Route distance
State_boundary = st_read("Data_input/dist_poly_new.shp") # State boundary
State_road_network = st_read("Data_input/USA_Route_final.shp") # State Road Network
Vulnerability   = read.csv("Data_input/dea_efficiencies_health_USA.csv") # State Vulnerability
Time_matrix        = read.csv("Data_input/mat_time.csv", row.names = 1) # state connections


Link_distance$Speed_SI_units = (Link_distance$dist_km*1000)/(Link_distance$time_hrs*60*60)
Link_distance$Speed_kmph = 18/5*Link_distance$Speed_SI_units
Link_distance$weight = NA


Link_distance$state_intermediate = 0
State_names = State_boundary$NAME


# Finding neighbouring State
row.names(State_boundary) <- as.character(State_boundary$NAME)
nb <- poly2nb(State_boundary) # finding neighbouring State
matrix <- nb2mat(nb, style="B",zero.policy = T) # creating matrix of neighbouring State
colnames(matrix) <- rownames(matrix)

matrix_link_wt = matrix 
count = 0


for(i in 1:length(State_names)){ # 'i' is row number 
  for(j in 1:i){                # 'j' is column number (to avoid duplications, 'j' is looped till 'i' number)
    matrix_link_wt[i,j] = matrix_link_wt[j,i] = 0 # initialization
    if(Time_matrix[i,j]>0){
      prt     = paste(State_names[i]," <-> ",State_names[j], sep = "")  ;  print(prt)
      prt_shp = paste(State_names[i],State_names[j], sep = "")     ;   print(prt_shp)
      nb_State = State_boundary$geometry[c(i,j)]                         ;   plot(nb_State)
      nb_link = State_road_network$geometry[State_road_network$Name_1 == prt_shp]   ;   plot(nb_link)
      nb_link = st_transform(nb_link, crs = st_crs(State_boundary)) # transforming the crs system
      tll     = st_length(nb_link) # tll = total link length
      link_i  = st_intersection(State_boundary$geometry[i], nb_link)    ;  # plot(link_i)
      link_j  = st_intersection(State_boundary$geometry[j], nb_link)    ;  # plot(link_j)
      l_i     = st_length(link_i)
      l_j     = st_length(link_j)
      
     
      
      HV_i = Vulnerability$ccr.eff[Vulnerability$health_data.NAME == State_names[i]]
      HV_j = Vulnerability$ccr.eff[Vulnerability$health_data.NAME == State_names[j]]
      
      weight = as.numeric((l_i * HV_i + l_j * HV_j) / tll)
      Link_distance$weight[Link_distance$i_name == State_names[i] & Link_distance$j_name == State_names[j]] = weight
      
      matrix_link_wt[i,j] = matrix_link_wt[j,i] = weight # matrix of link weights
      
      nb_check = (l_i+l_j)/tll # to check whether the link passes through state other than neighbouring state
      if(as.numeric(nb_check) < 0.95){
        nb_intermediate = st_crosses(nb_link,State_boundary)
        nb_int = unlist(nb_intermediate)
        
        if(length(nb_int) == 3){
          dist_1 = State_names[nb_int[1]]
          dist_2 = State_names[nb_int[2]]
          dist_3 = State_names[nb_int[3]]
          link_1 = st_intersection(State_boundary$geometry[nb_int[1]], nb_link) 
          link_2 = st_intersection(State_boundary$geometry[nb_int[2]], nb_link) 
          link_3 = st_intersection(State_boundary$geometry[nb_int[3]], nb_link)
          l1 = st_length(link_1)
          l2 = st_length(link_2)
          l3 = st_length(link_3)
          HV1 = Vulnerability$ccr.eff[Vulnerability$health_data.NAME == dist_1]
          HV2 = Vulnerability$ccr.eff[Vulnerability$health_data.NAME == dist_2]
          HV3 = Vulnerability$ccr.eff[Vulnerability$health_data.NAME == dist_3]
          
          weight = as.numeric((l1 * HV1 + l2 * HV2 + l3 * HV3 ) / (l1+l2+l3))
          Link_distance$weight[Link_distance$i_name == State_names[i] & Link_distance$j_name == State_names[j]] = weight
          matrix_link_wt[i,j] = matrix_link_wt[j,i] = weight
        }
        if(length(nb_int) == 4){
          dist_1 = State_names[nb_int[1]]
          dist_2 = State_names[nb_int[2]]
          dist_3 = State_names[nb_int[3]]
          dist_4 = State_names[nb_int[4]]
          link_1 = st_intersection(State_boundary$geometry[nb_int[1]], nb_link) 
          link_2 = st_intersection(State_boundary$geometry[nb_int[2]], nb_link) 
          link_3 = st_intersection(State_boundary$geometry[nb_int[3]], nb_link)
          link_4 = st_intersection(State_boundary$geometry[nb_int[4]], nb_link)
          l1 = st_length(link_1)
          l2 = st_length(link_2)
          l3 = st_length(link_3)
          l4 = st_length(link_4)
          HV1 = Vulnerability$ccr.eff[Vulnerability$health_data.NAME == dist_1]
          HV2 = Vulnerability$ccr.eff[Vulnerability$health_data.NAME == dist_2]
          HV3 = Vulnerability$ccr.eff[Vulnerability$health_data.NAME == dist_3]
          HV4 = Vulnerability$ccr.eff[Vulnerability$health_data.NAME == dist_4]
          
          weight = as.numeric((l1 * HV1 + l2 * HV2 + l3 * HV3 + l4 * HV4) / (l1+l2+l3+l4))
          Link_distance$weight[Link_distance$i_name == State_names[i] & Link_distance$j_name == State_names[j]] = weight
          matrix_link_wt[i,j] = matrix_link_wt[j,i] = weight
        }
        
        Link_distance$state_intermediate[Link_distance$i_name == State_names[i] & Link_distance$j_name == State_names[j]] = 1
        count = count + 1
      }  #else (stop("There is one more state between these neighbouring state shortest route"))
      
    }
  }
}   # ; print(count)
write.csv(Link_distance, file ="R_output/USA_weights_distance.csv", row.names = F )
write.csv(matrix_link_wt, file = "R_output/matrix_road_LINK_with_WEIGHTS.csv", row.names = T)
toc()
print(count)
