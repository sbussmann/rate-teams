
object RateTeamsBayesian extends App with StanPlot {

  val priorsDiv1 = List(0.0, 3.0, 5.0)
  val priorsDiv2 = List(0.0, 0.0, 0.0)
  val priorsDiv3 = List(0.0, -1.0, -3.0)
  val nChoices = priorsDiv1.size
  val allOutputs = for (iChoice <- 0 until nChoices) yield {
    val pD1 = priorsDiv1(iChoice)
    val pD2 = priorsDiv2(iChoice)
    val pD3 = priorsDiv3(iChoice)

    object Model1 extends ScalaStan {
      println(pD1, pD2, pD3)

      val N = data(int(lower = 0))
      val divMu = data(real()(N))
      val M = data(int(lower = 0))
      val teamAWon = data(int(lower = 0, upper = 1)(M))
      val teamsA = data(int(lower = 0)(M))
      val teamsB = data(int(lower = 0)(M))

      val source = CsvDataSource.fromString(scala.io.Source.fromResource("winloss_divprior_40264.csv").mkString)
      val wonBoolean: Seq[String] = source.readRaw("Team A Wins")
      val boolMap = Map(
        "True" -> 1,
        "False" -> 0
      )
      val wonData: Seq[Int] = wonBoolean.map(boolMap)
      val teamsAData: Seq[Int] = source.read(teamsA, "Index A").map(_ + 1)
      val teamsBData: Seq[Int] = source.read(teamsB, "Index B").map(_ + 1)
      val nTeamsData: Int = (teamsAData.toSet union teamsBData.toSet).size
      val nGamesData: Int = teamsAData.size

      val divAData: Seq[String] = source.readRaw("Div A")
      val divBData: Seq[String] = source.readRaw("Div B")

      // Average Div 1 team beats an average Div 2 team 95% of the time.
      // Average Div 2 team beats an average Div 3 team 73% of the time.
      val teamDivMap = Map(
        "4/3 Div 1" -> pD1,
        "4/3 Div 2" -> pD2,
        "4/3 Div 3" -> pD3,
        "5/2 Div 1" -> pD1,
        "5/2 Div 2" -> pD2,
        "5/2 Div 3" -> pD3
      )
      val muAData: Seq[Double] = divAData.map(teamDivMap)
      val muBData: Seq[Double] = divBData.map(teamDivMap)

      val skillVector = parameter(real()(N))
      //  val teamSkill = parameter(positiveOrdered(nTeams))
      val teamsMuData: Seq[Double] = for (teamIndex <- 1 to nTeamsData) yield {
        if (teamsAData.contains(teamIndex)) {
          val rightIndex = teamsAData.indexOf(teamIndex)
          muAData(rightIndex)
        } else {
          val rightIndex = teamsBData.indexOf(teamIndex)
          muBData(rightIndex)
        }
      }

      val model = new Model {
        for (team <- range(1, N)) {
          skillVector(team) ~ stan.normal(divMu(team), 1)
        }
        for (game <- range(1, M)) {
          val teamA = teamsA(game)
          val teamB = teamsB(game)
          val skillDelta = skillVector(teamA) - skillVector(teamB)
          teamAWon(game) ~ stan.bernoulli_logit(skillDelta)
        }

      }

      val results: StanResults = model
        .withData(teamAWon, wonData)
        .withData(teamsA, teamsAData)
        .withData(teamsB, teamsBData)
        .withData(N, nTeamsData)
        .withData(M, nGamesData)
        .withData(divMu, teamsMuData)
        .run(chains = 4, method = RunMethod.Sample(samples = 2000, warmup = 1000))

      //  val (meanResult, varianceResult) =
      val bestSkill: Seq[Double] = results.mean(skillVector)
    }
  }