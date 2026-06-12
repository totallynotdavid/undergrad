C  Seismic deformation model
C  Computation is based on the theory of Okada (1985)
C  Coded by Koshimura Shunichi 18 August,2000
C  Modified by C Jimenez 30 Jul 2014: output format: GMT or Matlab
C  Modified by C Jimenez 03 Set 2020: axes rotation to obtain: UE, UN
c  Modified by C Jimenez 22 Mar 2022: allocate and allocatable

C --- Variables for the output --- 
C     Z        : Surface displacement (m)
C
C --- Parameters required to be changed for each computation ---
C     IDS,IDE,JDS,JDE: Relative position of grid deformation on grid_a
C     IA,JA    : Grid Dimension of the computational domain
C     DX,DY    : Grid size (in meters)
C     NP       : Number of fault segments (NP=1 : Simple fault event)
C     SGL      : Parameter for definition of event type
C                SGL=0 : Single fault event, SGL=1 : Multiple fault event
C     D0       : Dislocation (m)
C     W0,L0    : Width,Length of the fault (m)
C     ST,DI,SL : Strike,Dip,Rake (deg)
C     HH       : Depth of upper fault edge (m)
C
C --- Parameters NOT (basically) required to be changed  ---
C
C     VP,VS    : P-wave and S-wave velocity (km/sec)
C     EPS      : Checking parameter for singular point call
C     RE       : Radius on the equator
C
C     Note     :
C     * I0 and J0 should be given by the grid number 
C       of the computational region (automatically computed in this code).
C  -------------------------------------------------
      INTEGER SGL
      REAL L0,str_rad
C  -------- You need to change parameters below -------
      PARAMETER (IDS= 153,IDE= 637,JDS= 844,JDE= 1396)
      PARAMETER (IA=IDE-IDS+1, JA=JDE-JDS+1)
      PARAMETER (DX=926.6244,DY=DX)
      PARAMETER (NP=1) 
C  -------- Correction Parameters for the Fault Position ---
C  -------- You don't have to change parameters below -------
      PARAMETER (VP=4.82E+3,VS=2.78E+3)
      PARAMETER (E2=0.006694470,RE=6377397.155)
      PARAMETER (EPS=1.0E-8)
C  -----------------------------------------------------------
      real(4), allocatable :: Z(:,:),UX(:,:),UY(:,:),UE(:,:),UN(:,:)
c     DIMENSION Z(IA,JA),UX(IA,JA),UY(IA,JA),UE(IA,JA),UN(IA,JA)
      DIMENSION I0(NP),J0(NP),D0(NP),HH(NP)
      DIMENSION L0(NP),W0(NP),ST(NP),DI(NP),SL(NP)
      DIMENSION DST(NP),DDP(NP)
C
      allocate(Z(IA,JA),UX(IA,JA),UY(IA,JA),UE(IA,JA),UN(IA,JA))
c     OPEN(28,FILE='dispE.grd',STATUS='UNKNOWN')
c     OPEN(29,FILE='dispN.grd',STATUS='UNKNOWN')
      OPEN(30,FILE='deform_a.grd',STATUS='UNKNOWN')
C
C  -------- You need to change parameters below -------
C     I0       : Position of each fault (or segment) in grid system 
C     J0       : Position of each fault (or segment) in grid system 
C     D0       : Dislocation of each fault (or segment) in meter
C     L0       : Fault length in strike direction in meter
C     W0       : Fault width in dip direction in meter
C     ST       : Strike angle
C     DI       : Dip angle
C     SL       : Rake angle
C     HH       : Depth of the fault origin in meter
C     
      OPEN(2,FILE='pfalla.inp',STATUS='OLD')
      DO N=1,NP
        READ(2,*)I0(N),J0(N),D0(N),L0(N),W0(N),ST(N),DI(N),SL(N),HH(N)
        IF (ST(N).EQ.0.0) ST(N)=ST(N)+0.001
        IF (ST(N).EQ.360.0) ST(N)=ST(N)+0.001
      END DO
      CLOSE(2)

C  ******************************
      PI=2.0*ASIN(1.0)
      RMU=VS*VS/(VP*VP-VS*VS)
C  ******************************
      IF (NP.EQ.1) SGL=0
      IF (NP.GT.1) SGL=1
C     SGL=0 : Single event
C     SGL=1 : Multiple event (Depending on the value of NP)
C
C     --- Correction of the Fault Position ---
      DO N=1,NP
         I0(N)=I0(N)-IDS+1
         J0(N)=J0(N)-JDS+1
      END DO
C
      DO N=1,NP
         WRITE(*,'(A20,I3,A3)')   ' Segment          : ',N,'-th'
         IF(SGL.EQ.0.AND.N.NE.1)GO TO 11 
         STR=(90.0-ST(N))*PI/180.0
         DIR=DI(N)*PI/180.0
         SLR=SL(N)*PI/180.0
C        
         CS=COS(DIR)
         SN=SIN(DIR)
         DE=HH(N)+W0(N)*SN
C
         DST(N)=D0(N)*COS(SLR)
         DDP(N)=D0(N)*SIN(SLR)
C
         DO J=JA,1,-1
         DO I=1,IA
         X0=((I-1)-I0(N))*DX
         Y0=((J-1)-J0(N))*DY
         X=X0/COS(STR)+(Y0-X0*TAN(STR))*SIN(STR)
         Y=Y0*COS(STR)-X0*SIN(STR)+W0(N)*CS
C
         P=Y*CS+DE*SN
         Q=Y*SN-DE*CS
C
         CALL USTRIKE
     &     (EPS,RMU,P,Q,CS,SN,L0(N),W0(N),X,P,FX1,FY1,FZ1) 
         CALL USTRIKE
     &     (EPS,RMU,P,Q,CS,SN,L0(N),W0(N),X,P-W0(N),FX2,FY2,FZ2) 
         CALL USTRIKE
     &     (EPS,RMU,P,Q,CS,SN,L0(N),W0(N),X-L0(N),P,FX3,FY3,FZ3) 
         CALL USTRIKE
     &     (EPS,RMU,P,Q,CS,SN,L0(N),W0(N),X-L0(N),P-W0(N),FX4,FY4,FZ4) 
C
         CALL UDIP
     &     (EPS,RMU,P,Q,CS,SN,L0(N),W0(N),X,P,GX1,GY1,GZ1) 
         CALL UDIP
     &     (EPS,RMU,P,Q,CS,SN,L0(N),W0(N),X,P-W0(N),GX2,GY2,GZ2) 
         CALL UDIP
     &     (EPS,RMU,P,Q,CS,SN,L0(N),W0(N),X-L0(N),P,GX3,GY3,GZ3) 
         CALL UDIP
     &     (EPS,RMU,P,Q,CS,SN,L0(N),W0(N),X-L0(N),P-W0(N),GX4,GY4,GZ4) 
C
         UXST=-(FX1-FX2-FX3+FX4)*DST(N)/(2.0*PI)
         UYST=-(FY1-FY2-FY3+FY4)*DST(N)/(2.0*PI)
         UZST=-(FZ1-FZ2-FZ3+FZ4)*DST(N)/(2.0*PI)
C
         UXDP=-(GX1-GX2-GX3+GX4)*DDP(N)/(2.0*PI)
         UYDP=-(GY1-GY2-GY3+GY4)*DDP(N)/(2.0*PI)
         UZDP=-(GZ1-GZ2-GZ3+GZ4)*DDP(N)/(2.0*PI)
C
         UX(I,J)=UX(I,J)+UXST+UXDP
         UY(I,J)=UY(I,J)+UYST+UYDP
         Z(I,J)=Z(I,J)+UZST+UZDP
         IF(ABS(Z(I,J)).GE.20.0)WRITE(*,*)I,J
C
         END DO
         END DO
      END DO
   11 CONTINUE
      CALL FEATURES (IA,JA,NP,I0,J0,D0,L0,W0,ST,DI,SL,HH,DST,DDP,Z,
     &               DX,DY,SGL)
C
      write (*,*) 
c     write (*,*) 'Escoger formato de salida:'
      write (*,*) '(1) Formato Matlab'
c     write (*,*) '(2) Formato GMT'
c     read (*,*) formato
      formato = 1
c Rotacion de ejes para obtener UE y UN
c      str_rad = ST(1)*PI/180.0;
c	  DO I=1,IA
c  	    DO J=1,JA
c            UE(I,J) = UX(I,J)*SIN(str_rad)-UY(I,J)*COS(str_rad)
c            UN(I,J) = UX(I,J)*COS(str_rad)+UY(I,J)*SIN(str_rad)
c	    END DO
c      END DO
	  
      if (formato.eq.1) then
      DO 10 I=1,IA
c        WRITE(28,22) (UE(I,J),J=1,JA)
c        WRITE(29,22) (UN(I,J),J=1,JA)
         WRITE(30,22) ( Z(I,J),J=1,JA)
10    CONTINUE
      
      else
c     WRITE(28,'(10F7.3)')((UE(I,J),I=1,IA),J=JA,1,-1)
c     WRITE(29,'(10F7.3)')((UN(I,J),I=1,IA),J=JA,1,-1)
c Escribir la cabecera
      write(30,'(a,I7)') 'ncols', IA
      write(30,'(a,I7)') 'nrows', JA
      write(30,'(a,F7.2)') 'xllcorner', -79.00
      write(30,'(a,F7.2)') 'yllcorner', -20.00
      write(30,'(a,1F10.6)') 'cellsize', DX/1000.0/111.1994
      write(30,'(a)') 'nodata_value -9999'
c Fin de escribir la cabecera
      WRITE(30,'(10F7.3)')(( Z(I,J),I=1,IA),J=JA,1,-1)
      end if

22    FORMAT(4000F7.3)
      CLOSE(28)
      CLOSE(29)
      CLOSE(30)
	  deallocate(Z,UX,UY,UE,UN)
      STOP
      END
C
      SUBROUTINE USTRIKE (EPS,RMU,P,Q,CS,SN,L0,W0,XI,ET,FX,FY,FZ)
C
      REAL L0
C
      YH=ET*CS+Q*SN
      DH=ET*SN-Q*CS
      R=SQRT(XI*XI+ET*ET+Q*Q)
      RET=R+ET
      RDH=R+DH
      XX=SQRT(XI*XI+Q*Q)
C
C     *** Call of singular point (R+ET=0) ***
      IF(ABS(RET).GE.EPS)THEN
C
C        In the regular points
C
C        *** Call of singular point ( Cos(DIP)=0 ) ***
         IF(ABS(CS).GE.EPS)THEN
            XI4=RMU*(LOG(R+DH)-SN*LOG(R+ET))/CS
            XI3=RMU*(YH/(CS*(R+DH))-LOG(R+ET))+XI4*SN/CS
C           *** Call of singular point (XI=0) ***
            IF(ABS(XI).GE.EPS)THEN
               XI5IN=(ET*(XX+Q*CS)+XX*(R+XX)*SN)/(XI*(R+XX)*CS)
               XI5=RMU*2.0*(ATAN(XI5IN))/CS
            ELSE
               XI5=0.0
            END IF
            XI1=RMU*(-XI/(CS*(R+DH)))-XI5*SN/CS
            XI2=RMU*(-LOG(R+ET))-XI3
         ELSE
            XI1=-RMU*XI*Q/(2.0*(R+DH)*(R+DH))
            XI3=(RMU/2.0)*(ET/(R+DH)+YH*Q/((R+DH)*(R+DH))-LOG(R+ET))
            XI4=-RMU*Q/(R+DH)
C           *** Call of singular point (XI=0) ***
            IF(ABS(XI).GE.EPS)THEN
               XI5=-RMU*XI*SN/(R+DH)
            ELSE
               XI5=0.0
            END IF
            XI2=RMU*(-LOG(R+ET))-XI3
         END IF
C
      ELSE
C
C        In the singular points
C
C        *** Call of singular point ( Cos(DIP)=0 ) ***
         IF(ABS(CS).GE.EPS)THEN
C
C           *** Call of singular point (R+DH=0) ***
            IF(ABS(RDH).GE.EPS)THEN
               XI4=RMU*(LOG(R+DH)+SN*LOG(R-ET))/CS
               XI3=RMU*(YH/(CS*(R+DH))+LOG(R-ET))+XI4*SN/CS
C              *** Call of singular point (XI=0) ***
               IF(ABS(XI).GE.EPS)THEN
                  XI5IN=(ET*(XX+Q*CS)+XX*(R+XX)*SN)/(XI*(R+XX)*CS)
                  XI5=RMU*2.0*(ATAN(XI5IN))/CS
               ELSE
                  XI5=0.0
               END IF
               XI1=RMU*(-XI/(CS*(R+DH)))-XI5*SN/CS
               XI2=RMU*(LOG(R-ET))-XI3
            ELSE
               XI4=RMU*(-LOG(R-DH)+SN*LOG(R-ET))/CS
               XI3=RMU*(LOG(R-ET))+XI4*SN/CS
C              *** Call of singular point (XI=0) ***
               IF(ABS(XI).GE.EPS)THEN
                  XI5IN=(ET*(XX+Q*CS)+XX*(R+XX)*SN)/(XI*(R+XX)*CS)
                  XI5=RMU*2.0*(ATAN(XI5IN))/CS
               ELSE
                  XI5=0.0
               END IF
               XI1=-XI5*SN/CS
               XI2=RMU*(LOG(R-ET))-XI3
            END IF
C
         ELSE
C
C           *** Call of singular point (R+DH=0) ***
            IF(ABS(RDH).GE.EPS)THEN
               XI1=-RMU*XI*Q/(2.0*(R+DH)*(R+DH))
               XI3=(RMU/2.0)*(ET/(R+DH)+YH*Q/((R+DH)*(R+DH))+LOG(R-ET))
               XI4=-RMU*Q/(R+DH)
C              *** Call of singular point (XI=0) ***
               IF(ABS(XI).GE.EPS)THEN
                  XI5=-RMU*XI*SN/(R+DH)
               ELSE
                  XI5=0.0
               END IF
               XI2=RMU*(LOG(R-ET))-XI3
            ELSE
               XI1=0.0
	       XI3=0.0
	       XI4=0.0
	       XI5=0.0
               XI2=RMU*(LOG(R-ET))-XI3
            END IF
         END IF
      END IF
C     *** End ***
C
C
C     *** Call of singular point (R+ET=0) ***
      IF(ABS(RET).GE.EPS)THEN
C
C        In the regular points
C
         SUX1=XI*Q/(R*(R+ET))
C        *** Call of singular point (Q=0) ***
         IF(ABS(Q).GE.EPS)THEN
            SUX2=ATAN(XI*ET/(Q*R))
         ELSE
            SUX2=0.0
         END IF
         SUX3=XI1*SN
         FX=SUX1+SUX2+SUX3
C
         SUY1=YH*Q/(R*(R+ET))
         SUY2=Q*CS/(R+ET)
         SUY3=XI2*SN
         FY=SUY1+SUY2+SUY3
C
         SUZ1=DH*Q/(R*(R+ET))
         SUZ2=Q*SN/(R+ET)
         SUZ3=XI4*SN
         FZ=SUZ1+SUZ2+SUZ3
C
      ELSE
C
C        In the singular points (R+ET=0)
C
         SUX1=0.0
C        *** Call of singular point (Q=0) ***
         IF(ABS(Q).GE.EPS)THEN
            SUX2=ATAN(XI*ET/(Q*R))
         ELSE
            SUX2=0.0
         END IF
         SUX3=XI1*SN
         FX=SUX1+SUX2+SUX3
C
         SUY1=0.0
         SUY2=0.0
         SUY3=XI2*SN
         FY=SUY1+SUY2+SUY3
C
         SUZ1=0.0
         SUZ2=0.0
         SUZ3=XI4*SN
         FZ=SUZ1+SUZ2+SUZ3
C
      END IF
C     *** End ***
      IF(ABS(FZ).GE.20.0)WRITE(*,*)Q,R+DH,RET,EPS
      RETURN
      END
C
      SUBROUTINE UDIP (EPS,RMU,P,Q,CS,SN,L0,W0,XI,ET,GX,GY,GZ)
C
      REAL L0
C
      YH=ET*CS+Q*SN
      DH=ET*SN-Q*CS
      R=SQRT(XI*XI+ET*ET+Q*Q)
      RET=R+ET
      XX=SQRT(XI*XI+Q*Q)
C
C     *** Call of singular point (R+ET=0) ***
      IF(ABS(RET).GE.EPS)THEN
C
C        In the regular points
C
C        *** Call of singular point ( Cos(DIP)=0 ) ***
         IF(ABS(CS).GE.EPS)THEN
            XI4=RMU*(LOG(R+DH)-SN*LOG(R+ET))/CS
            XI3=RMU*(YH/(CS*(R+DH))-LOG(R+ET))+XI4*SN/CS
C           *** Call of singular point (XI=0) ***
            IF(ABS(XI).GE.EPS)THEN
               XI5IN=(ET*(XX+Q*CS)+XX*(R+XX)*SN)/(XI*(R+XX)*CS)
               XI5=RMU*2.0*(ATAN(XI5IN))/CS
            ELSE
               XI5=0.0
            END IF
            XI1=RMU*(-XI/(CS*(R+DH)))-XI5*SN/CS
            XI2=RMU*(-LOG(R+ET))-XI3
         ELSE
            XI1=-RMU*XI*Q/(2.0*(R+DH)*(R+DH))
            XI3=(RMU/2.0)*(ET/(R+DH)+YH*Q/((R+DH)*(R+DH))-LOG(R+ET))
            XI4=-RMU*Q/(R+DH)
C           *** Call of singular point (XI=0) ***
            IF(ABS(XI).GE.EPS)THEN
               XI5=-RMU*XI*SN/(R+DH)
            ELSE
               XI5=0.0
            END IF
            XI2=RMU*(-LOG(R+ET))-XI3
         END IF
C
      ELSE
C
C        In the singular points
C
C        *** Call of singular point ( Cos(DIP)=0 ) ***
         IF(ABS(CS).GE.EPS)THEN
            XI4=RMU*(LOG(R+DH)+SN*LOG(R-ET))/CS
            XI3=RMU*(YH/(CS*(R+DH))+LOG(R-ET))+XI4*SN/CS
C            *** Call of singular point (XI=0) ***
            IF(ABS(XI).GE.EPS)THEN
               XI5IN=(ET*(XX+Q*CS)+XX*(R+XX)*SN)/(XI*(R+XX)*CS)
               XI5=RMU*2.0*(ATAN(XI5IN))/CS
            ELSE
               XI5=0.0
            END IF
            XI1=RMU*(-XI/(CS*(R+DH)))-XI5*SN/CS
            XI2=RMU*(LOG(R-ET))-XI3
         ELSE
            XI1=-RMU*XI*Q/(2.0*(R+DH)*(R+DH))
            XI3=(RMU/2.0)*(ET/(R+DH)+YH*Q/((R+DH)*(R+DH))+LOG(R-ET))
            XI4=-RMU*Q/(R+DH)
C           *** Call of singular point (XI=0) ***
            IF(ABS(XI).GE.EPS)THEN
               XI5=-RMU*XI*SN/(R+DH)
            ELSE
               XI5=0.0
            END IF
            XI2=RMU*(LOG(R-ET))-XI3
         END IF
C
      END IF
C     *** End ***
C
      UX1=Q/R
      UX2=-XI3*SN*CS
      GX=UX1+UX2
C
C     *** Call of singular point (R+XI=0) ***
      IF(ABS(R+XI).GE.EPS)THEN
         UY1=YH*Q/(R*(R+XI)) 
      ELSE
         UY1=0.0
      END IF
C     *** Call of singular point (Q=0) ***
      IF(ABS(Q).GE.EPS)THEN
         UY2=CS*ATAN(XI*ET/(Q*R))
      ELSE
         UY2=0.0
      END IF
      UY3=-XI1*SN*CS
      GY=UY1+UY2+UY3
C
C     *** Call of singular point (R+XI=0) ***
      IF(ABS(R+XI).GE.EPS)THEN
         UZ1=DH*Q/(R*(R+XI))
      ELSE
         UZ1=0.0
      END IF
C     *** Call of singular point (Q=0) ***
      IF(ABS(Q).GE.EPS)THEN
         UZ2=SN*ATAN(XI*ET/(Q*R))
        ELSE
         UZ2=0.0
      END IF
      UZ3=-XI5*SN*CS
      GZ=UZ1+UZ2+UZ3
C
      IF(ABS(GZ).GE.20.0)WRITE(*,*)R+XI,R+DH,RET,EPS
      RETURN
      END
C
      SUBROUTINE FEATURES 
     &           (IA,JA,NP,I0,J0,D0,L0,W0,ST,DI,SL,HH,DST,DDP,Z,
     &            DX,DY,SGL)
C
      INTEGER SGL
      REAL L0
      DIMENSION Z(IA,JA)
      DIMENSION I0(NP),J0(NP),D0(NP),HH(NP)
      DIMENSION L0(NP),W0(NP),ST(NP),DI(NP),SL(NP)
      DIMENSION DST(NP),DDP(NP)
C
      WRITE(*,'(A45)')'******** Features of the grid system ********'
      WRITE(*,'(A45,F9.4,A1,F9.4)')
     &                ' Grid size DX, DY     (meter)              : ',
     &      DX,',',DY
      WRITE(*,'(A45)')'******** Features of the fault model ********'
      IF(SGL.EQ.0)WRITE(*,'(A29)')'This is a single fault event.' 
      IF(SGL.EQ.1)WRITE(*,'(A30)')'This is multiple faults event.' 
      DO N=1,NP
      IF(SGL.EQ.0.AND.N.NE.1)GO TO 22 
      WRITE(*,'(A45)')'---------------------------------------------'
c     WRITE(*,'(A20,I2,A3)')   ' Segment          : ',N,'-th'
      WRITE(*,'(A10,I5,A7,I5)')' Segmento  : ',N,'-th de ',NP
      WRITE(*,'(A20,I4,A1,I4)')' Fault origin     : ',I0(N),',',J0(N)
      WRITE(*,'(A20,F5.1)')    ' Fault Length  (km):',L0(N)/1000.0
      WRITE(*,'(A20,F5.1)')    ' Fault Width   (km):',W0(N)/1000.0
      WRITE(*,'(A20,F5.1)')    ' Strike       (deg):',ST(N)
      WRITE(*,'(A20,F5.1)')    ' Dip          (deg):',DI(N)
      WRITE(*,'(A20,F5.1)')    ' Rake         (deg):',SL(N)
      WRITE(*,'(A20,F5.1)')    ' Depth         (km):',HH(N)/1000.0
      WRITE(*,'(A20,F6.2)')    ' Dislocation    (m):',D0(N)
      WRITE(*,'(A24,F6.2)')    '  Strike-slip component : ',DST(N)
      WRITE(*,'(A24,F6.2)')    '  Dip-slip    component : ',DDP(N)
      END DO
   22 CONTINUE
      ZMAX=-100.0
      ZMIN=100.0
      DO J=JA,1,-1
         DO I=1,IA
            IF(ZMAX.LT.Z(I,J))ZMAX=Z(I,J)
            IF(ZMIN.GT.Z(I,J))ZMIN=Z(I,J)
         END DO
      END DO
      WRITE(*,'(A45)')'---------------------------------------------'
      WRITE(*,'(A45)')'******** Features of the fault model ********'
      WRITE(6,'(A23,F6.2)')'Maximum uplift    (m): ',ZMAX
      WRITE(6,'(A23,F6.2)')'Maximum subsidence(m): ',ZMIN
C
      RETURN
      END
C
